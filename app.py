
import gradio as gr
from fastai.vision.all import *
from pathlib import Path

# Ana dizindeki veri yolu
path = Path('.')

# Görsel sayısını logla
print("Bulunan resim sayısı:", len(get_image_files(path)))

# Sabit sınıf listesi
vocab = ['accordion', 'banjo', 'drum', 'flute', 'guitar', 'harmonica', 'saxophone', 'sitar', 'tabla', 'violin']

# DataLoaders oluştur
def get_dls():
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=vocab)),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=Resize(460),
        batch_tfms=aug_transforms(size=224)
    )
    return dblock.dataloaders(path, bs=64)

# Modeli yükle
dls = get_dls()
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.load('music_weights')

# Tahmin fonksiyonu
def predict(img):
    pred_class, pred_idx, probs = learn.predict(img)
    max_prob = float(probs[pred_idx])

    if max_prob < 0.3:
        return "Bu görselin net bir şekilde bir müzik aleti olduğunu anlayamadım."

    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Gradio arayüzü
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Müzik Aleti Tanıma ",
    description="Bir müzik aleti görseli yükleyin, model tahmin etsin."
)

interface.launch()
