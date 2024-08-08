# DGMs_SoC-Project

Generative Adversarial Network (GAN)
One of the most powerful AI technologies in development today is generative adversarial networks, which encompass a relatively new way for machines to learn and create that are leading to highly successful results.

Architecture and training process
GANs are adversarial in nature and involve a game between two deep learning submodels called the generator and discriminator. 


The generator learns to create fake data that resembles the original domain data. The discriminator learns to distinguish between the fake data from the generator and the real data. At first, the discriminator can easily tell the two sets of data apart. As training progresses and the models make adjustments according to the results, the generator improves until the discriminator struggles to easily distinguish the fake from the real data. Through these iterations, GANs achieve a level of realism and authenticity in its output that can fool the human senses, such as videos of destinations that look like real places or photographs of corgis in beachwear that appear to be real dogs.

Applications
GANs are typically employed for imagery or visual data, including image generation, image enhancement, video predictions and style transfer.

Strengths
GANs excel at generating high-quality and realistic content, particularly when it comes to images.

Weaknesses
GANs have been known to be difficult to train due to instability in the interactions of the two submodels. The generator and discriminator can fail to reach an optimal equilibrium or state of convergence, oscillating in their abilities to outperform each other. This instability can lead to mode collapse, which means the generator learns to only create a limited subset of samples from the target distribution rather than the entire distribution. For example, a GAN trained to create cat images may start creating only orange tabby cat images. This limitation in generated samples means a degeneration in the quality and diversity of the output data.

Variational Autoencoder (VAE)
The second prominent generative model in use today is variational autoencoders. VAEs are a deep generative model that, similarly to GANs, rely on two neural networks to generate data. Traditionally, VAEs work to compress and reconstruct data, which is useful for tasks such as cleaning data and reducing the dimensionality of data sets to, say, improve the performance of an algorithm.

Architecture and training process
The dual networks, called encoders and decoders, work in tandem to generate an output that is similar to the input. 


The encoder compresses the input data (into what’s called the latent space) to optimize for the most efficient representation of the original data while retaining only the most important information. The decoder then reconstructs the input from the compressed representation. The decoder in this way generates content and is able to achieve a high-level of detail to generate specific features.

Applications
VAEs are great for cleaning noise from images and finding anomalies in data. They are also flexible and customizable to specific tasks compared to other approaches.Today, they are used for anything from image generation to anomaly detection such as in fraud detection for financial institutions.

Strengths
VAEs learn a probabilistic distribution over latent space, allowing for quantifying uncertainty in data and anomaly detection. They are also easier to train and more stable than GANs.

Weaknesses
A weakness of VAEs is they tend to produce lower quality content, such as blurry images, compared to other methods like GANs. They also struggle to capture complex and highly structured data.
