pass
from PIL import Image, ImageDraw

def render(data, labels):
    image = Image.new('RGB', (1000, 650), 'white')
    draw = ImageDraw.Draw(image)
    boxes = []
    draw.line((100, 90, 100, 500, 920, 500), fill='#40574a', width=3)
    boxes.append(put(image, labels['title'], 500, 40, 32, max_width=900))
    for (i, (key, value)) in enumerate(zip(data['categories'], data['values'])):
        x = 240 + i * 320
        height = value / 10 * 360
        draw.rectangle((x - 65, 500 - height, x + 65, 500), fill='#4e8870')
        boxes.append(put(image, str(value), x, 480 - height, 28))
        boxes.append(put(image, labels[key], x, 550, 30, max_width=280))
    return (image, boxes)
