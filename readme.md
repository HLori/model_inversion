# Requirements

- pytorch 1.5.1
- PIL 

# Structure

- `data_facescrub`:  aligned faces for training. 200 person
  - `Aaron_Eckhart`: images of this person
- `data2_facescrub`: aligned faces, person who do not occur in training dataset. 63 person
- `adv_pert`: adversarial perturbation, original data is in `data2_facescrub`, target class is in `data_facescrubt`
  - `0_Aaron_Eckhart`: target_Name Of Target
    - `Matthew_Perry`: name of original data.
      - `0_from_Matthew_Perry.jpg`: idx_from_original name
- `adv_example`: adversarial examples.

- `adversarial.py` : generate adversarial examples.
  - `data_path`:  original image path in test data. eg. `./data2_facescrub`
  - `dest_path`: generated adversarial example. eg. `./adv_example`
  - `dest2`: generated adversarial perturbation. eg. `./adv_pert`
  - `model_path`: target model's path. eg. `./model/face_weights.pt`
- `train.py`: train a target model.
  - store in `./model/face_weights.pt`: 
    - model: `ResNet18`
    - train set: *facescrub*
    - test accuracy: 75%
- `ResNet.py`: definition of models.
  - `ResNet18`: ResNet18
  - `UnetGenerator`:  Generator of DCGAN with UNET, using in pix2pix
  - `NLayerDiscriminator`：Discriminator of DCGAN,
    - input: n\*6\*h*w,  `torch.cat((img, adv), dim=1)`
  - `CGAN`:  GAN structure without $z$ distribution
    - `self.save(path, epoch)`: saving model
    - `self.load(path)`: load model
    - `self.setrequires_grad(nets, requires_grad)`: set netD or netG requiring grads or not
    - `self.optimize_parameters(input, real)`:  optimize both D and G with input and real
    - `self.get_output(input)`: get G(input)

