run("Gaussian Blur...", "sigma=0.80 stack");
run("Enhance Contrast...", "saturated=0 process_all");
run("3D Objects Counter", "threshold=140 slice=90 min.=10 max.=4865280 exclude_objects_on_edges objects summary");
