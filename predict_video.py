import cv2
from ultralytics import YOLO
import argparse




def predict_video(model, input_video_path, output_video_path):

    video = cv2.VideoCapture(input_video_path)

    # параметри відео
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter( output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height) )

    # обробляємо кожен кадр
    while video.isOpened():

        success, frame = video.read()

        if not success:
            break

        # виявляємо птахів
        results = model(frame)

        # малюємо bounding boxes, назви класів і confidence
        annotated_frame = results[0].plot()

        # записуємо кадр у вихідне відео
        writer.write(annotated_frame)

    # закриваємо вхідний та вихідний відеофайли
    video.release()
    writer.release()

    print(f'Відео успішно збережено в {output_video_path}')




def main():

    print('Завантаження моделі та підготовка до обробки відео..')

    parser = argparse.ArgumentParser(description="Bird detection on video")

    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output video")

    args = parser.parse_args()

    weights_path = 'weights/best.pt'
    model = model = YOLO(weights_path)  

    predict_video(model, args.input, args.output)


    # тестування на прикладах
    #for i in range(1, 7):
    
        #input_video_path = f'examples/example{i}/input.mp4'
        #output_video_path = f'examples/example{i}/output.mp4'
    
         #predict_video(model, input_video_path, output_video_path)




if __name__ == "__main__":
    main()
