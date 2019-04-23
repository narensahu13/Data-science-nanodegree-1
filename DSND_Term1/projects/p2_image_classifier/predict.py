# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 23:59:17 2018

@author: Narendra.Sahu
"""

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    im = Image.open(image_path)
    processed_im = process_image(im).unsqueeze(0)
    model.to(device)
    model.eval()    
    with torch.no_grad():
        processed_im = processed_im.to('cuda').float()
        output = model(processed_im)
        ps = torch.exp(output)
    pred = ps.topk(topk)
    flower_ids = pred[1][0].to('cpu')
    flower_ids = torch.Tensor.numpy(flower_ids)
    probs = pred[0][0].to('cpu')
    idx_to_class = {k:v for v,k in checkpoint['class_to_idx'].items()}
    flower_names = np.array([cat_to_name[idx_to_class[x]] for x in flower_ids])
        
    return probs, flower_names

import matplotlib.image as mpimg
# TODO: Display an image along with the top 5 classes
image_path = 'flowers/test/28/image_05230.jpg'
im = process_image(Image.open(image_path))

probs, flower_names = predict(image_path, model)

fig, ax = plt.subplots(2,figsize=(5,10))
image = Image.open(image_path)
tranfs = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
ax[0].imshow(tranfs(image))
ax[1].barh(flower_names, probs)