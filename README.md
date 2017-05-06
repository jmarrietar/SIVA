Evaluation of <i> Weakly Supervised Learning</i> paradigms on automatic visual inspection
=================================================================================

Overview
--------

SIVA contains a Python Code implementations created for use in the following publication:
> Arrieta, Mera and Espinosa. **Evaluation of weakly supervised learning paradigms
> on automatic visual inspection.** _To appear in
> 

Abstract
--------
<i> In computer vision, supervised learning has been used to model the relationships between the characteristics of a group of objects (or patterns) and their class labels. This model is constructed based on a set of training images that have been tagged by an expert in the problem domain. In general, this labelling process consists of defining the object's class tag as well as demarcating the precise area of ​​the image in which it is located. In many computer vision applications, such as automatic visual inspection, this labelling process can be a labour intensive task requiring extensive work and may even become impractical when there are many images of training. This limitation has led to the development of new learning algorithms that allow some ambiguity or "weakness" in the way class tags are assigned. These algorithms are known as weakly supervised learning algorithms. In this paper we present an evaluation of different weakly supervised learning paradigms grouped according to the representation of the object of interest (in one or several characteristic vectors) associated to its class label (single or multiple label). Experimentation is performed on a set of texture images, adding an additional artificial defect. The experimental results show that using a representation of the object in multiple instances, presents better results in the identification of defective image. </i>



Weak learning and object-label relationship
--------------------------------------------

<div align="center">
<img src="https://github.com/jmarrietar/SIVA/blob/master/img/Slide1.jpg" width="50%" height="50%"/>
</div>
<br>

> **Z. H. Zhou, M. L. Zhang, S. J. Huang, and Y. F. Li, “Multi-instance multi-label learning,” Artif. Intell., vol. 176, no. 1, > pp. 2291–2320, 2012.**


Experimentation
--------------

#### Additional Defect

An additional defect to the original image is necessary in order to use learning paradigms that use multiple labels. For this reason an artificial defect is added to the images in the 5 kinds of textures. The additional defect is created using a medium smoothing filter where where the center element is replaced by the median of all pixels under the area of ​​the kernel used. The parameter used for classes 1 and 5 are 9 pixels, for classes 2 and 3 are 11 pixels and for class 4 a size of 13 pixels is used.

#### Instance Generation

For generation of instances in the weak learning paradigms, simple instances as in multiple instances, we used a strategy based on weak labeling (Weak labeling). Here, the expert defines a region enclosing the defect of the image and neighboring regions. A block size of 32px x 32px is slid through the area defined by the expert and extracts the characteristics in each block of the region enclosing the defect. These zones were automatically defined with help of the ground truth of the images.

<div align="center">
<img src="https://github.com/jmarrietar/SIVA/blob/master/img/collage2.jpeg" width="50%" height="50%"/>
</div>
<br>

Original Images at https://hci.iwr.uni-heidelberg.de/node/3616

Conclusions
-----------
In this paper, a literature review of the weak learning paradigms grouped according to the representation of the object of interest and its class label was carried out. This theoretical revision was complemented by a practical application in the context of automatic visual inspection, where different kinds of textures were chosen and defects identified using different paradigms of weak learning.

It is concluded that in the automatic visual inspection of defects in texture images, it is a good approximation the representation of the images in the Multi-Instance-Single-Label (MISL) learning paradigm, because it allows to successfully model the images as Bags with defects and bags without defects. Additionally, the computation time in the training process in this paradigm is more favorable in comparison to the other groups, so the continuous learning of an automatic visual inspection system would be more agile using this paradigm in comparison to the others.

Based on the above, the future work in this area of ​​research should be focused on the application of weak learning paradigms in different contexts of image pattern recognition where the ambiguity in the representation of the object of interest is present and the application Of weak learning paradigms may be useful.
