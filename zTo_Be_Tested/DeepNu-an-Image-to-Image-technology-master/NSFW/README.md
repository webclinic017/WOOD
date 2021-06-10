# NSFW
NSFW(Not Safe/Suitable For Work) is a large-scale image dataset containing five categories of images [porn, hentai, sexy, natural, drawings]. Here, CycleGAN is used to convert different types of images, such as porn->natural. 使用CycleGAN神经网络模型实现 [porn, hentai, sexy, natural, drawings] 这些类别图像的转换，比如色情图像到安全中性的图像转换。

|NSFW image category|English|中文|
|-|-|-|
|Drawing|Harmless art, or picture of art|无害的艺术图画，包括动漫|
|Hentai |Pornographic art, unsuitable for most work environments|色情艺术图，不适合大多数工作环境|
|Neutral|General, inoffensive content|安全、中性图片|
|Porn|Indecent content and actions, often involving genitalia| 色情图片，性行为，通常涉及生殖器|
|Sexy |Unseemly provocative content, can include nipples|性感图片，而非色情图片，包括乳头|


## NSFWJS detector demo
> Learn more clicks [nsfwjs](https://github.com/infinitered/nsfwjs)

This section provides a demo erotic image detector nsfwjs, which will determine the type of image you are uploading (porn, hentai, sexy, natural, drawings), as shown below.

这一部分提供一个试玩的色情图片检测器nsfwjs，该检测器会确定你上传图片的类别（porn, hentai, sexy, natural, drawings），如下图所示

![](NSFW_JS_demo.png)

[Try NSFWJS Demo](https://nsfwjs.com/)


## NSFW dataset

+ [nsfw_data_scraper](https://github.com/alex000kim/nsfw_data_scraper) Collection of scripts to aggregate image data for the purposes of training an NSFW Image Classifier
+ [nsfw_data_source_urls](https://github.com/EBazarov/nsfw_data_source_urls) Collection of NSFW images URLs for the purposes of training an NSFW Image Classifier

[中文报道](https://www.jiqizhixin.com/articles/2019-02-14-7)


## NSFW Detection Machine Learning Model

+ [nsfw_model](https://github.com/GantMan/nsfw_model) Keras model of NSFW detector


## NSFW_GAN

1. First download the data set
2. Neural network model come soon
