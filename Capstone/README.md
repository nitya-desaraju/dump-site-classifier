# Illegal Waste Dump Site Identifier

This was created as a final project for BWSI's Cog*works. It uses resnets and other machine learning models to identify illegal waste dump sites and what waste is in it. This can improve city planning and the environment.

## How this works

We used an existing Resnet 50 model in order to generate a heat map of where the waste is. Then we create a bounding box for it to then feed it into a Resnet 18. We fine-tuned the resnet's last layer to have it identify the waste type.

## Challenges

The biggest problem was the dataset. It didn't have good amounts of data in a way that was useful. For example, the dataset was large, but there would only be a small handful of images for a certain waste classification, meaning that the model would be poorly trained for that. So, we narrowed down what our model did and augmented the data by rotating images to increase the size of the dataset. We first used a resnet 50 for the waste classification, but it was overfitting the data, so we switched to resnet 18. Finally, we reached 90% accuracy. We learned a lot about data preprocessing and the theory behind residual networks. After the model was complete, we had difficulty running the model on a backend server so the results could be displayed on the website because the model was trained on cuda and we couldn't easily switch. We ended up just doing the demo with cuda.

[![Athena Award Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Faward.athena.hackclub.com%2Fapi%2Fbadge)](https://award.athena.hackclub.com?utm_source=readme)