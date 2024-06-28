run("Gaussian Blur...", "sigma=2 stack");
run("3D Objects Counter", "threshold=35 slice=328 min.=50 max.=172134000 exclude_objects_on_edges objects surfaces statistics");
selectImage("Surface map of Data_3d_view.tif");

