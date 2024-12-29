**Optix Raytracer**

# Introduce
This program is a simple ray tracer based on OptiX API.

<img width="1920" alt="image0" src="https://github.com/user-attachments/assets/2a633ff4-8550-43dd-a3cd-a15ae29b658d" />
<img width="1920" alt="image1" src="https://github.com/user-attachments/assets/84d67acd-7ee6-4223-989a-a93ee37c5f60" />
<img width="1920" alt="image2" src="https://github.com/user-attachments/assets/42778998-fa93-4eb4-b5c1-268f93b89b14" />
<img width="1920" alt="image4" src="https://github.com/user-attachments/assets/68cb31ae-0d31-4694-b52a-76404967d058" />
<img width="1920" alt="image5" src="https://github.com/user-attachments/assets/c6982ab0-1a90-40ca-b22d-10eb6c0685db" />

# Contents
### Geometric Primitives
- cube and cylinder of OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES
- polygon model of OPTIX_BUILD_INPUT_TYPE_TRIANGLES
- point cloud data set of OPTIX_BUILD_INPUT_TYPE_SPHERES

### Scene composition
- 1 Instance Acceleration Structure(IAS) of 4 Geometry Acceleration Structures(GAS)

### Toon shading
- simple toon shading application to polygon models

# Reference
optixWhitted project from OptiX SDK 8.1.0 samples.

<https://developer.nvidia.com/designworks/optix/download>
