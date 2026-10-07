// Solid hyperelastic ball for the Section 4.6 impact study.
// Units: mm. The ball is meshed centred on the origin; build_ball_case.py
// translates it so that its lowest point sits just above the rigid floor.
SetFactory("OpenCASCADE");

If (!Exists(radius))
  radius = 25.0;
EndIf
If (!Exists(lc))
  lc = 3.0;
EndIf

Sphere(1) = {0, 0, 0, radius};

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Physical Volume(1) = {1};
