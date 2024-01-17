from ultralytics import YOLO
import numpy as np


model = YOLO('./runs/classify/train4/weights/last.pt')  # load a custom model


results = model('D:\Github\Accident-and-Damage-Detection-Identifying-and-Categorizing-Involved-Vehicles\img_accident.jpg')  # predict on an image



names_dict = results[0].names

probs = results[0].probs.data.tolist()

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])

if names_dict[np.argmax(probs)] == 'severe':
    print('Chances of occurring fatality very high. Sending the information.....')
elif names_dict[np.argmax(probs)] == 'moderate':
    print('Chances of occurring fatality is quite possible, taking initiative.....')
elif names_dict[np.argmax(probs)] == 'minor':
    print('Chance are very low. take your own step')
