from config import RANDOM_SEED
from ultralytics import YOLO
import os
import shutil




def main():

    model = YOLO("yolo26n.pt")

    results = model.train(
        data="data/processed/data.yaml",
        epochs=100,
        imgsz=800,
        batch=8,
        workers=2,
        patience=15,
        name="bird_detector",
        seed=RANDOM_SEED,
        deterministic=True,
        device=0, 

        # розширення даних 
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        scale=0.3,
        translate=0.1,
        degrees=5.0
    )


    # створення папок для збереження результатів навчання
    os.makedirs('./weights', exist_ok=True)
    os.makedirs('./training_results', exist_ok=True)

    # шлях до результатів навчання
    run_dir = results.save_dir

    # зберігаємо найкращі ваги в потрібну папку
    shutil.copy2( f'{run_dir}/weights/best.pt', f'./weights/best.pt' )

    # зберігаємо CSV з результатами навчання в потрібну папку
    shutil.copy2( f'{run_dir}/results.csv', f'./training_results/results.csv' )




if __name__ == "__main__":
    main()