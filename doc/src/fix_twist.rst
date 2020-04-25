.. index:: fix twist

fix twist command
====================

Syntax
""""""


.. code-block:: LAMMPS

   fix ID group-ID twist keyword args ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* twist = style name of this fix command
* one or more keyword/arg pairs may be appended
* keyword = *torque* or *constraint* or *torque_changing* or *per_turn* or *per_turn_changing*
  
  .. parsed-literal::
  
       *torque* args = atom1 atom2 atom3 atom4 tau
         atom1,atom2,atom3,atom4 = IDs of 4 atoms in dihedral in linear order
         tau = torque (in units of energy/radian) applied to these 4 atoms.
  
       *constraint* args = atom1 atom2 atom3 atom4 k phi_a phi_b
         atom1,atom2,atom3,atom4 = IDs of 4 atoms in dihedral in linear order
         k = harmonic spring constant (in units of energy per radian^2)
         phi_a = rest angle at the beginning of the simulation
         phi_b = rest angle at the end of the simulation

       *torque_changing* args = atom1 atom2 atom3 atom4 tau_a tau_b
         atom1,atom2,atom3,atom4 = IDs of 4 atoms in dihedral in linear order
         tau_a = torque applied at the beginning of the run
         tau_b = torque applied at the end of the run
  
       *per_turn* args = atom1 atom2 atom3 atom4 tau
         atom1,atom2,atom3,atom4 = IDs of 4 atoms in dihedral in linear order
         tau = torque (in units of energy/revolution) applied to these 4 atoms.

       *per_turn_changing* args = atom1 atom2 atom3 atom4 tau_a tau_b
         atom1,atom2,atom3,atom4 = IDs of 4 atoms in dihedral in linear order
         tau_a = torque (in units of energy/revolution) applied at the beginning of the run
         tau_b = torque (in units of energy/revolution) applied at the end of the run


Examples
""""""""
.. code-block:: LAMMPS

   # Apply a constant torque (1.5 energy units per radian)
   fix TWIST all twist torque 1001 1002 1003 1004 1.50

   # Apply a constant torque in units of energy units per 360 degree turn
   fix TWIST all twist per_turn 1001 1002 1003 1004 10.0

   # Apply a torque that increases from 0 to 1.5 over the simulation run
   fix TWIST all twist torque_changing 1001 1002 1003 1004 0.0 1.50

   # Apply a torque that increases from 0.0 to 10.0 (energy per 360 turn)
   fix TWIST all twist per_turn_changing 1001 1002 1003 1004 0.0 10.0

   # Apply a harmonic constraint (spring constant 20.0) that forces the atoms
   # to rotate around their central axis at a constant rate, increasing the
   # torsion angle from -60 to 7800 degrees during the simulation run
   fix TWIST all twist velocity 1001 1002 1003 1004 20.0 -60.0 7800.0

   # Apply a constant torque to atoms 1 2 3 4, AND force
   # atoms 3 4 5 6 to spin around their central axis at a constant rate
   fix TWIST all twist torque 1 2 3 4 1.50 constraint 3 4 5 6 20.0 -60 7800.0


Description
"""""""""""

Apply one or more torsional forces which encourage 4 atoms to rotate around
their central axis (defined by the axis passing through the two interior atoms).
This fix applies forces to a specified sets of atoms by making them part
of a dihedral interaction whose strength or direction can vary over time
during a simulation.
This is functionally similar to creating dihedral for the same atoms in a data
file, as specified by the :doc:`read_data <read_data>` command, albeit
with a time-varying pre-factor coefficient, and except for exclusion
rules, as explained below.

Use this fix to temporarily force atoms to twist around an axis.
Alternatively, you can use the :doc:`dihedral_style table <dihedral_table>`
command (along with the "linear" keyword) to apply this torque permanently.

The group-ID specified by this fix is ignored.

.. note::

   This fix applies a force to the system which, in general,
   does *not* conserve energy.
   However it is possible to define an energy for *some* systems
   (for example, twisted circular polymers under constant torsional tension), 
   For dynamics via the :doc:`run <run>` command,
   this energy can be added to the system's
   potential energy for thermodynamic output (see below).

.. note::

   Adding a fix with this command does not apply
   the exclusion rules and weighting factors specified by the
   :doc:`special_bonds <special_bonds>` command to atoms in the fix.
   that are now bonded (1-4 neighbors) as a result.
   The forces from this fix will be applied to these atoms independently,
   on top of other forces acting on them.  This is true even if they are close
   enough to participate in non-bonded (:doc:`pair_style <pair_style>`)
   interactions.

----------

The *torque* keyword applies a constant torsional force
that exerts a torque on the specified atoms around their central axis,
with the following parameters:

* :math:`\tau` (energy per radian)

:math:`\tau` is specified with the fix.

Note: As an alternative to using *fix twist*, you can instead
use the :doc:`dihedral_style table <dihedral_table>` command
(along with the "linear" keyword) to apply permanent
torques to atoms that appear in the *Dihedrals* section of a LAMMPS data file.

----------

The *torque_changing* keyword applies a torque on the specified atoms
which varies over the simulation run from :math:`\tau_a` to :math:`\tau_b`.

* :math:`\tau_a` (energy per radian)
* :math:`\tau_b` (energy per radian)

----------

Alternatively, the *per_turn* and *per_turn_changing* keywords behave
in an identical way to the *torque* and *torque_changing* keywords.
However when they are used, the :math:`\tau` parameters are specified
in units of energy-per-turn (360-degree turn), instead of energy-per-radian.

----------

The *constrain* keyword applies a time-varying harmonic torsional constraint
to the specified atoms.  This constraint forces the atoms to twist at a constant
rate around their central axis.  The potential associated with this force is:

.. math::

   E = (k/2) (\phi - \phi_0(t))^2
   \phi_0 = \phi_a + (\phi_b-\phi_a)(t/T)

with the following parameters:

* :math:`k` (energy)
* :math:`phi_a` (degrees)
* :math:`phi_b` (degrees)

Here, :math:`T` denotes the *duration* of the upcoming simulation run
and :math:`t` denotes the current time relative the the start of the run.

The :math:`k`, :math:`\phi_a`, and :math:`\phi_b` parameters are specified
with the fix.
Note that the :math:`phi_a` and :math:`phi_b` parameters *need not*
lie in the range from :math:`0` to :math:`2\pi`.
Note also that in many harmonic forcefields in LAMMPS, the usual factor of 1/2
is absorbed into the value of :math:`k`.  That is *not* true in this case.

----------


**Restart, fix\_modify, output, run start/stop, minimize info:**

No information about this fix is written to :doc:`binary restart files <restart>`.

The :doc:`fix_modify <fix_modify>` *energy* option is supported by this
fix to add the potential energy associated with this fix to the
system's potential energy as part of :doc:`thermodynamic output <thermo_style>`.

The :doc:`fix_modify <fix_modify>` *respa* option is supported by this
fix. This allows to set at which level of the :doc:`r-RESPA <run_style>`
integrator the fix is adding its forces. Default is the outermost level.
(WARNING: This feature was inherited from "fix restraint"
and has not been tested.)

.. note::

   If you want the fictitious potential energy associated with the
   added forces to be included in the total potential energy of the
   system (the quantity being minimized), you MUST enable the
   :doc:`fix_modify <fix_modify>` *energy* option for this fix.
   (WARNING: This feature has not been tested carefully.)
   Note again that this fix does *not conserve energy*
   (except in some limited circumstances).

This fix computes a global scalar which can be accessed by various
:doc:`output commands <Howto_output>`.
The scalar is the total potential energy for *all* the torsion interactions
discussed above.  The energy calculated by this fix is "extensive".
(WARNING: This feature was inherited from "fix restraint"
and has not been tested.)

Restrictions
""""""""""""
 none

**Related commands:** :doc:`dihedral_style table <dihedral table>`, :doc:`fix_modify <fix_modify>`

**Default:** none
