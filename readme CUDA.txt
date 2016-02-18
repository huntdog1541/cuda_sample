Compilation and Debugging Instructions for CUDA programs on Tesla
-----------------------------------------------------------------

Requirements: Source code must have extension .cu.


Compile code:

Run nvcc the cuda compiler with a specified cuda source code file.

    nvcc vecdot.cu -o vecdot 

This compiles vecdot.cu into an executable vecdot that runs on host and device.
To generate debug symbols, use:

	nvcc -g -G vecdot.cu -o vecdot

Debug code:

To step through the code use cuda-gdb:

	cuda-gdb vecdot

Set breakpoints in main and the function running on the cuda device:

break main
break vecDot

Also set a break point on a line 

Run the code by entering run.

Step through the code using (n)ext, (c) continue.

You can print out values of variables on cuda device and on host using print.



