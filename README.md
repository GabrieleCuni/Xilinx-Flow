# Xilinx-Flow
My thesis aimed to create a stable flow for implementing deep neural networks on a Xilinx FPGA board using Python and the Xilinx Vitis AI environment. The developed software takes a state-of-the-art convolutional network for image recognition as input and performs the necessary transformations to generate an executable neural model for the FPGA board.

With the development flow in place, the study focused on exploring the achievable performance in terms of accuracy and speed of inference. This evaluation involved testing 16 different Mobilenets on 8 different hardware architectures. The hardware architectures for FPGA configuration were created using the Xilinx DPU-PYNQ software.

To conduct the tests, the Xilinx Zynq Ultrascale+ MPSoC board equipped with the PYNQ operating system was used. The testing script was also written in Python, and hardware acceleration was achieved by leveraging the libraries provided by PYNQ.
