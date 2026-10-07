// Five-hole silicone-rubber specimen reconstructed from the measured
// coordinates reported in Fig. 3(f) of Meunier et al. (2008).
// Units are millimetres.
SetFactory("OpenCASCADE");

thickness = 1.75;
holeRadius = 10.0;
cutWidth = 1.5;
DefineConstant[
  meshSizeMin = {1.20, Name "Mesh/minimum size"},
  meshSizeMax = {2.50, Name "Mesh/maximum size"},
  curvaturePoints = {24, Name "Mesh/curvature points"}
];

Point(1) = {  0.0,   0.0, 0}; // O
Point(2) = {-62.0,   0.0, 0}; // B
Point(3) = {-61.5, -82.5, 0}; // C
Point(4) = {  0.0, -82.5, 0}; // D
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Disk(10) = {-47.5, -21.2, 0, holeRadius, holeRadius};
Disk(11) = {-14.0, -23.0, 0, holeRadius, holeRadius};
Disk(12) = {-31.5, -40.5, 0, holeRadius, holeRadius};
Disk(13) = {-47.5, -59.0, 0, holeRadius, holeRadius};
Disk(14) = {-14.5, -58.0, 0, holeRadius, holeRadius};

cutDx = -31.5 - (-47.5);
cutDy = -40.5 - (-21.2);
cutLength = Sqrt(cutDx * cutDx + cutDy * cutDy);
cutAngle = Atan2(cutDy, cutDx);
Rectangle(20) = {-47.5, -21.2 - cutWidth / 2, 0,
                 cutLength, cutWidth};
Rotate {{0, 0, 1}, {-47.5, -21.2, 0}, cutAngle} { Surface{20}; }

planarSpecimen() = BooleanDifference{ Surface{1}; Delete; }{
  Surface{10, 11, 12, 13, 14, 20}; Delete;
};
// Use two structured layers through the thickness.  Extruding the triangular
// surface mesh with Recombine creates six-node prisms and avoids the highly
// irregular interior tetrahedra that inverted in the large-displacement run.
extruded[] = Extrude {0, 0, thickness} {
  Surface{planarSpecimen()}; Layers{2}; Recombine;
};
Physical Volume("RUBBER", 1) = {extruded[1]};

Mesh.MshFileVersion = 2.2;
Mesh.ElementOrder = 1;
Mesh.MeshSizeMin = meshSizeMin;
Mesh.MeshSizeMax = meshSizeMax;
Mesh.MeshSizeFromCurvature = curvaturePoints;
Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.SaveAll = 1;
