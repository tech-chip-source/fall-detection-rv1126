cd ~/rknn_model_zoo/examples/yolov8/cpp
rm -rf build
mkdir build
cd build
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DTARGET_SOC=rv1126
make -j4
