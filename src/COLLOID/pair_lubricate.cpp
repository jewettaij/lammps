// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Randy Schunk (SNL)
                         Amit Kumar and Michael Bybee (UIUC)
------------------------------------------------------------------------- */

#include "pair_lubricate.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fix_deform.h"
#include "fix_wall.h"
#include "force.h"
#include "input.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "variable.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairLubricate::PairLubricate(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;

  // set comm size needed by this Pair

  comm_forward = 6;
}

/* ---------------------------------------------------------------------- */

PairLubricate::~PairLubricate()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(cut_inner);
  }
}

/* ---------------------------------------------------------------------- */

void PairLubricate::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fx,fy,fz,tx,ty,tz;
  double rsq,r,h_sep,radi;
  double vr1,vr2,vr3,vnnr,vn1,vn2,vn3;
  double vt1,vt2,vt3,wt1,wt2,wt3,wdotn;
  double vRS0;
  double vi[3],vj[3],wi[3],wj[3],xl[3];
  double a_sq,a_sh,a_pu;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double lamda[3],vstream[3];

  double vxmu2f = force->vxmu2f;

  ev_init(eflag,vflag);

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // subtract streaming component of velocity, omega, angmom
  // assume fluid streaming velocity = box deformation rate
  // vstream = (ux,uy,uz)
  // ux = h_rate[0]*x + h_rate[5]*y + h_rate[4]*z
  // uy = h_rate[1]*y + h_rate[3]*z
  // uz = h_rate[2]*z
  // omega_new = omega - curl(vstream)/2
  // angmom_new = angmom - I*curl(vstream)/2
  // Ef = (grad(vstream) + (grad(vstream))^T) / 2

  if (shearing) {
    double *h_rate = domain->h_rate;
    double *h_ratelo = domain->h_ratelo;

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];
      radi = radius[i];

      domain->x2lamda(x[i],lamda);
      vstream[0] = h_rate[0]*lamda[0] + h_rate[5]*lamda[1] +
        h_rate[4]*lamda[2] + h_ratelo[0];
      vstream[1] = h_rate[1]*lamda[1] + h_rate[3]*lamda[2] + h_ratelo[1];
      vstream[2] = h_rate[2]*lamda[2] + h_ratelo[2];
      v[i][0] -= vstream[0];
      v[i][1] -= vstream[1];
      v[i][2] -= vstream[2];

      omega[i][0] += 0.5*h_rate[3];
      omega[i][1] -= 0.5*h_rate[4];
      omega[i][2] += 0.5*h_rate[5];
    }

    // set Ef from h_rate in strain units

    Ef[0][0] = h_rate[0]/domain->xprd;
    Ef[1][1] = h_rate[1]/domain->yprd;
    Ef[2][2] = h_rate[2]/domain->zprd;
    Ef[0][1] = Ef[1][0] = 0.5 * h_rate[5]/domain->yprd;
    Ef[0][2] = Ef[2][0] = 0.5 * h_rate[4]/domain->zprd;
    Ef[1][2] = Ef[2][1] = 0.5 * h_rate[3]/domain->zprd;

    // copy updated velocity/omega/angmom to the ghost particles
    // no need to do this if not shearing since comm->ghost_velocity is set

    comm->forward_comm(this);
  }

  // This section of code adjusts R0/RT0/RS0 if necessary due to changes
  // in the volume fraction as a result of fix deform or moving walls

  double dims[3], wallcoord;
  if (flagVF) // Flag for volume fraction corrections
    if (flagdeform || flagwall == 2) { // Possible changes in volume fraction
      if (flagdeform && !flagwall)
        for (j = 0; j < 3; j++)
          dims[j] = domain->prd[j];
      else if (flagwall == 2 || (flagdeform && flagwall == 1)) {
         double wallhi[3], walllo[3];
         for (int j = 0; j < 3; j++) {
           wallhi[j] = domain->prd[j];
           walllo[j] = 0;
         }
         for (int m = 0; m < wallfix->nwall; m++) {
           int dim = wallfix->wallwhich[m] / 2;
           int side = wallfix->wallwhich[m] % 2;
           if (wallfix->xstyle[m] == FixWall::VARIABLE) {
             wallcoord = input->variable->compute_equal(wallfix->xindex[m]);
           }
           else wallcoord = wallfix->coord0[m];
           if (side == 0) walllo[dim] = wallcoord;
           else wallhi[dim] = wallcoord;
         }
         for (int j = 0; j < 3; j++)
           dims[j] = wallhi[j] - walllo[j];
      }
      double vol_T = dims[0]*dims[1]*dims[2];
      double vol_f = vol_P/vol_T;
      if (flaglog == 0) {
        R0  = 6*MY_PI*mu*rad*(1.0 + 2.16*vol_f);
        RT0 = 8*MY_PI*mu*pow(rad,3.0);
        RS0 = 20.0/3.0*MY_PI*mu*pow(rad,3.0)*
          (1.0 + 3.33*vol_f + 2.80*vol_f*vol_f);
      } else {
        R0  = 6*MY_PI*mu*rad*(1.0 + 2.725*vol_f - 6.583*vol_f*vol_f);
        RT0 = 8*MY_PI*mu*pow(rad,3.0)*(1.0 + 0.749*vol_f - 2.469*vol_f*vol_f);
        RS0 = 20.0/3.0*MY_PI*mu*pow(rad,3.0)*
          (1.0 + 3.64*vol_f - 6.95*vol_f*vol_f);
      }
    }

  // end of R0 adjustment code

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    radi = radius[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // angular velocity

    wi[0] = omega[i][0];
    wi[1] = omega[i][1];
    wi[2] = omega[i][2];

    // FLD contribution to force and torque due to isotropic terms
    // FLD contribution to stress from isotropic RS0

    if (flagfld) {
      f[i][0] -= vxmu2f*R0*v[i][0];
      f[i][1] -= vxmu2f*R0*v[i][1];
      f[i][2] -= vxmu2f*R0*v[i][2];
      torque[i][0] -= vxmu2f*RT0*wi[0];
      torque[i][1] -= vxmu2f*RT0*wi[1];
      torque[i][2] -= vxmu2f*RT0*wi[2];

      if (shearing && vflag_either) {
        vRS0 = -vxmu2f * RS0;
        v_tally_tensor(i,i,nlocal,newton_pair,
                       vRS0*Ef[0][0],vRS0*Ef[1][1],vRS0*Ef[2][2],
                       vRS0*Ef[0][1],vRS0*Ef[0][2],vRS0*Ef[1][2]);
      }
    }

    if (!flagHI) continue;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);

        // angular momentum = I*omega = 2/5 * M*R^2 * omega

        wj[0] = omega[j][0];
        wj[1] = omega[j][1];
        wj[2] = omega[j][2];

        // xl = point of closest approach on particle i from its center

        xl[0] = -delx/r*radi;
        xl[1] = -dely/r*radi;
        xl[2] = -delz/r*radi;

        // velocity at the point of closest approach on both particles
        // v = v + omega_cross_xl - Ef.xl

        // particle i

        vi[0] = v[i][0] + (wi[1]*xl[2] - wi[2]*xl[1])
                        - (Ef[0][0]*xl[0] + Ef[0][1]*xl[1] + Ef[0][2]*xl[2]);

        vi[1] = v[i][1] + (wi[2]*xl[0] - wi[0]*xl[2])
                        - (Ef[1][0]*xl[0] + Ef[1][1]*xl[1] + Ef[1][2]*xl[2]);

        vi[2] = v[i][2] + (wi[0]*xl[1] - wi[1]*xl[0])
                        - (Ef[2][0]*xl[0] + Ef[2][1]*xl[1] + Ef[2][2]*xl[2]);

        // particle j

        vj[0] = v[j][0] - (wj[1]*xl[2] - wj[2]*xl[1])
                        + (Ef[0][0]*xl[0] + Ef[0][1]*xl[1] + Ef[0][2]*xl[2]);

        vj[1] = v[j][1] - (wj[2]*xl[0] - wj[0]*xl[2])
                        + (Ef[1][0]*xl[0] + Ef[1][1]*xl[1] + Ef[1][2]*xl[2]);

        vj[2] = v[j][2] - (wj[0]*xl[1] - wj[1]*xl[0])
                        + (Ef[2][0]*xl[0] + Ef[2][1]*xl[1] + Ef[2][2]*xl[2]);

        // scalar resistances XA and YA

        h_sep = r - 2.0*radi;

        // if less than the minimum gap use the minimum gap instead

        if (r < cut_inner[itype][jtype])
          h_sep = cut_inner[itype][jtype] - 2.0*radi;

        // scale h_sep by radi

        h_sep = h_sep/radi;

        // scalar resistances

        if (flaglog) {
          a_sq = 6.0*MY_PI*mu*radi*(1.0/4.0/h_sep + 9.0/40.0*log(1.0/h_sep));
          a_sh = 6.0*MY_PI*mu*radi*(1.0/6.0*log(1.0/h_sep));
          a_pu = 8.0*MY_PI*mu*pow(radi,3.0)*(3.0/160.0*log(1.0/h_sep));
        } else
          a_sq = 6.0*MY_PI*mu*radi*(1.0/4.0/h_sep);

        // relative velocity at the point of closest approach
        // includes fluid velocity

        vr1 = vi[0] - vj[0];
        vr2 = vi[1] - vj[1];
        vr3 = vi[2] - vj[2];

        // normal component (vr.n)n

        vnnr = (vr1*delx + vr2*dely + vr3*delz)/r;
        vn1 = vnnr*delx/r;
        vn2 = vnnr*dely/r;
        vn3 = vnnr*delz/r;

        // tangential component vr - (vr.n)n

        vt1 = vr1 - vn1;
        vt2 = vr2 - vn2;
        vt3 = vr3 - vn3;

        // force due to squeeze type motion

        fx  = a_sq*vn1;
        fy  = a_sq*vn2;
        fz  = a_sq*vn3;

        // force due to all shear kind of motions

        if (flaglog) {
          fx = fx + a_sh*vt1;
          fy = fy + a_sh*vt2;
          fz = fz + a_sh*vt3;
        }

        // scale forces for appropriate units

        fx *= vxmu2f;
        fy *= vxmu2f;
        fz *= vxmu2f;

        // add to total force

        f[i][0] -= fx;
        f[i][1] -= fy;
        f[i][2] -= fz;

        if (newton_pair || j < nlocal) {
          f[j][0] += fx;
          f[j][1] += fy;
          f[j][2] += fz;
        }

        // torque due to this force

        if (flaglog) {
          tx = xl[1]*fz - xl[2]*fy;
          ty = xl[2]*fx - xl[0]*fz;
          tz = xl[0]*fy - xl[1]*fx;

          torque[i][0] -= vxmu2f*tx;
          torque[i][1] -= vxmu2f*ty;
          torque[i][2] -= vxmu2f*tz;

          if (newton_pair || j < nlocal) {
            torque[j][0] -= vxmu2f*tx;
            torque[j][1] -= vxmu2f*ty;
            torque[j][2] -= vxmu2f*tz;
          }

          // torque due to a_pu

          wdotn = ((wi[0]-wj[0])*delx + (wi[1]-wj[1])*dely +
                   (wi[2]-wj[2])*delz)/r;
          wt1 = (wi[0]-wj[0]) - wdotn*delx/r;
          wt2 = (wi[1]-wj[1]) - wdotn*dely/r;
          wt3 = (wi[2]-wj[2]) - wdotn*delz/r;

          tx = a_pu*wt1;
          ty = a_pu*wt2;
          tz = a_pu*wt3;

          torque[i][0] -= vxmu2f*tx;
          torque[i][1] -= vxmu2f*ty;
          torque[i][2] -= vxmu2f*tz;

          if (newton_pair || j < nlocal) {
            torque[j][0] += vxmu2f*tx;
            torque[j][1] += vxmu2f*ty;
            torque[j][2] += vxmu2f*tz;
          }
        }

        if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
                                 0.0,0.0,-fx,-fy,-fz,delx,dely,delz);
      }
    }
  }

  // restore streaming component of velocity, omega, angmom

  if (shearing) {
    double *h_rate = domain->h_rate;
    double *h_ratelo = domain->h_ratelo;

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];
      radi = radius[i];

      domain->x2lamda(x[i],lamda);
      vstream[0] = h_rate[0]*lamda[0] + h_rate[5]*lamda[1] +
        h_rate[4]*lamda[2] + h_ratelo[0];
      vstream[1] = h_rate[1]*lamda[1] + h_rate[3]*lamda[2] + h_ratelo[1];
      vstream[2] = h_rate[2]*lamda[2] + h_ratelo[2];
      v[i][0] += vstream[0];
      v[i][1] += vstream[1];
      v[i][2] += vstream[2];

      omega[i][0] -= 0.5*h_rate[3];
      omega[i][1] += 0.5*h_rate[4];
      omega[i][2] -= 0.5*h_rate[5];
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLubricate::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(cut_inner,n+1,n+1,"pair:cut_inner");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLubricate::settings(int narg, char **arg)
{
  if (narg != 5 && narg != 7) error->all(FLERR,"Illegal pair_style command");

  mu = utils::numeric(FLERR,arg[0],false,lmp);
  flaglog = utils::inumeric(FLERR,arg[1],false,lmp);
  flagfld = utils::inumeric(FLERR,arg[2],false,lmp);
  cut_inner_global = utils::numeric(FLERR,arg[3],false,lmp);
  cut_global = utils::numeric(FLERR,arg[4],false,lmp);

  flagHI = flagVF = 1;
  if (narg == 7) {
    flagHI = utils::inumeric(FLERR,arg[5],false,lmp);
    flagVF = utils::inumeric(FLERR,arg[6],false,lmp);
  }

  if (flaglog == 1 && flagHI == 0) {
    error->warning(FLERR,"Cannot include log terms without 1/r terms; "
                   "setting flagHI to 1");
    flagHI = 1;
  }

  // reset cutoffs that have been explicitly set

  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) {
          cut_inner[i][j] = cut_inner_global;
          cut[i][j] = cut_global;
        }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLubricate::coeff(int narg, char **arg)
{
  if (narg != 2 && narg != 4)
    error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));

  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double cut_inner_one = cut_inner_global;
  double cut_one = cut_global;
  if (narg == 4) {
    cut_inner_one = utils::numeric(FLERR,arg[2],false,lmp);
    cut_one = utils::numeric(FLERR,arg[3],false,lmp);
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut_inner[i][j] = cut_inner_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLubricate::init_style()
{
  if (!atom->omega_flag)
    error->all(FLERR,"Pair lubricate requires atom attribute omega");
  if (!atom->radius_flag)
    error->all(FLERR,"Pair lubricate requires atom attribute radius");
  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair lubricate requires ghost atoms store velocity");

  neighbor->add_request(this);

  // require that atom radii are identical within each type
  // require monodisperse system with same radii for all types

  double radtype;
  for (int i = 1; i <= atom->ntypes; i++) {
    if (!atom->radius_consistency(i,radtype))
      error->all(FLERR,"Pair lubricate requires monodisperse particles");
    if (i > 1 && radtype != rad)
      error->all(FLERR,"Pair lubricate requires monodisperse particles");
    rad = radtype;
  }

  // check for fix deform, if exists it must use "remap v"
  // If box will change volume, set appropriate flag so that volume
  // and v.f. corrections are re-calculated at every step.
  //
  // If available volume is different from box volume
  // due to walls, set volume appropriately; if walls will
  // move, set appropriate flag so that volume and v.f. corrections
  // are re-calculated at every step.

  shearing = flagdeform = flagwall = 0;

  auto fixes = modify->get_fix_by_style("^deform");
  if (fixes.size() > 0) {
    shearing = flagdeform = 1;
    auto *myfix = dynamic_cast<FixDeform *>(fixes[0]);
    if (myfix && (myfix->remapflag != Domain::V_REMAP))
      error->all(FLERR,"Using pair lubricate with inconsistent fix deform remap option");
  }
  fixes = modify->get_fix_by_style("^wall");
  if (fixes.size() > 1)
    error->all(FLERR, "Cannot use multiple fix wall commands with pair lubricate");
  else if (fixes.size() == 1) {
    wallfix = dynamic_cast<FixWall *>(fixes[0]);
    if (!wallfix)
      error->all(FLERR, "Fix {} is not compatible with pair lubricate", fixes[0]->style);
    flagwall = 1;
    if (wallfix->xflag) flagwall = 2; // Moving walls exist
  }

  // set the isotropic constants that depend on the volume fraction
  // vol_T = total volume

  double vol_T;
  double wallcoord;
  if (!flagwall) vol_T = domain->xprd*domain->yprd*domain->zprd;
  else {
    double wallhi[3], walllo[3];
    for (int j = 0; j < 3; j++) {
      wallhi[j] = domain->prd[j];
      walllo[j] = 0;
    }

    for (int m = 0; m < wallfix->nwall; m++) {
      int dim = wallfix->wallwhich[m] / 2;
      int side = wallfix->wallwhich[m] % 2;
      if (wallfix->xstyle[m] == FixWall::VARIABLE) {
        wallfix->xindex[m] = input->variable->find(wallfix->xstr[m]);
        //Since fix->wall->init happens after pair->init_style
        wallcoord = input->variable->compute_equal(wallfix->xindex[m]);
      }

      else wallcoord = wallfix->coord0[m];

      if (side == 0) walllo[dim] = wallcoord;
      else wallhi[dim] = wallcoord;
    }
    vol_T = (wallhi[0] - walllo[0]) * (wallhi[1] - walllo[1]) *
      (wallhi[2] - walllo[2]);
  }

  // vol_P = volume of particles, assuming monodispersity
  // vol_f = volume fraction

  vol_P = atom->natoms*(4.0/3.0)*MY_PI*pow(rad,3.0);
  double vol_f = vol_P/vol_T;

  if (!flagVF) vol_f = 0;

  // set isotropic constants for FLD

  if (flaglog == 0) {
    R0  = 6*MY_PI*mu*rad*(1.0 + 2.16*vol_f);
    RT0 = 8*MY_PI*mu*pow(rad,3.0);
    RS0 = 20.0/3.0*MY_PI*mu*pow(rad,3.0)*(1.0 + 3.33*vol_f + 2.80*vol_f*vol_f);
  } else {
    R0  = 6*MY_PI*mu*rad*(1.0 + 2.725*vol_f - 6.583*vol_f*vol_f);
    RT0 = 8*MY_PI*mu*pow(rad,3.0)*(1.0 + 0.749*vol_f - 2.469*vol_f*vol_f);
    RS0 = 20.0/3.0*MY_PI*mu*pow(rad,3.0)*(1.0 + 3.64*vol_f - 6.95*vol_f*vol_f);
  }


  // set Ef = 0 since used whether shearing or not

  Ef[0][0] = Ef[0][1] = Ef[0][2] = 0.0;
  Ef[1][0] = Ef[1][1] = Ef[1][2] = 0.0;
  Ef[2][0] = Ef[2][1] = Ef[2][2] = 0.0;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLubricate::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    cut_inner[i][j] = mix_distance(cut_inner[i][i],cut_inner[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  cut_inner[j][i] = cut_inner[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLubricate::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&cut_inner[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLubricate::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&cut_inner[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&cut_inner[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLubricate::write_restart_settings(FILE *fp)
{
  fwrite(&mu,sizeof(double),1,fp);
  fwrite(&flaglog,sizeof(int),1,fp);
  fwrite(&flagfld,sizeof(int),1,fp);
  fwrite(&cut_inner_global,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&flagHI,sizeof(int),1,fp);
  fwrite(&flagVF,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLubricate::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR,&mu,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&flaglog,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&flagfld,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_inner_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&flagHI,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&flagVF,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&mu,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&flaglog,1,MPI_INT,0,world);
  MPI_Bcast(&flagfld,1,MPI_INT,0,world);
  MPI_Bcast(&cut_inner_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&flagHI,1,MPI_INT,0,world);
  MPI_Bcast(&flagVF,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

int PairLubricate::pack_forward_comm(int n, int *list, double *buf,
                                     int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;

  double **v = atom->v;
  double **omega = atom->omega;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = v[j][0];
    buf[m++] = v[j][1];
    buf[m++] = v[j][2];
    buf[m++] = omega[j][0];
    buf[m++] = omega[j][1];
    buf[m++] = omega[j][2];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void PairLubricate::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  double **v = atom->v;
  double **omega = atom->omega;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
    omega[i][0] = buf[m++];
    omega[i][1] = buf[m++];
    omega[i][2] = buf[m++];
  }
}

/* ----------------------------------------------------------------------
   check if name is recognized, return integer index for that name
   if name not recognized, return -1
   if type pair setting, return -2 if no type pairs are set
------------------------------------------------------------------------- */

int PairLubricate::pre_adapt(char *name, int /*ilo*/, int /*ihi*/, int /*jlo*/, int /*jhi*/)
{
  if (strcmp(name,"mu") == 0) return 0;
  return -1;
}

/* ----------------------------------------------------------------------
   adapt parameter indexed by which
   change all pair variables affected by the reset parameter
   if type pair setting, set I-J and J-I coeffs
------------------------------------------------------------------------- */

void PairLubricate::adapt(int /*which*/, int /*ilo*/, int /*ihi*/, int /*jlo*/, int /*jhi*/,
                          double value)
{
  mu = value;
}
