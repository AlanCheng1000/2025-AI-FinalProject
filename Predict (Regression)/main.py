import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import shuffle
from loguru import logger
from geopy.distance import geodesic
import pandas as pd
import os

from CNN import CNN, train, validate, test
from utils import TrainDataset, TestDataset, load_train_dataset, load_test_dataset

def main(): 
    target = 'mag' # 預測目標: mag/dep/loc/lon/lat
    nRound = 10 # 執行次數
    avg_mae = 0
    EPOCHS = 10
    lr = 1e-4
    model_type = 'resnet18' # 圖片使用模型: resnet18/efficientnet_b0/simple_cnn
    use_extra_params = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss() # L1Loss/SmoothL1Loss/MSELoss

    for i in range(nRound):
        logger.info(f"[Round {i + 1}/{nRound} - {target}]")
        if target == 'loc':
            """
            Lon
            """
            subtarget = 'lon'
            logger.info("Start loading data")
            images, labels, extra_params = load_train_dataset(subtarget)
            images, labels, extra_params = shuffle(images, labels, extra_params, random_state=777)
            train_len = int(0.9 * len(images))

            train_images, val_images = images[:train_len], images[train_len:]
            train_labels, val_labels = labels[:train_len], labels[train_len:]
            train_extra_params, val_extra_params = extra_params[:train_len], extra_params[train_len:]
            test_images, test_extra_params = load_test_dataset(subtarget)

            train_dataset = TrainDataset(train_images, train_labels, train_extra_params)
            val_dataset = TrainDataset(val_images, val_labels, val_extra_params)
            test_dataset = TestDataset(test_images, test_extra_params)
            
            logger.info("Start training CNN")
            model = CNN(model_type, use_extra_params).to(device)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Optimizer configuration
            base_params = [param for name, param in model.named_parameters() if param.requires_grad]
            optimizer = optim.Adam(base_params, lr=lr)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            min_MAE = 1000000

            for epoch in range(EPOCHS): #epoch
                train_loss = train(model, train_loader, criterion, optimizer, device)
                val_loss, MAE = validate(model, val_loader, criterion, device)
                scheduler.step(val_loss)

                if MAE < min_MAE:
                    min_MAE = MAE
                    torch.save(model.state_dict(), f'best_model({subtarget}).pth')
                logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | MAE: {MAE:.4f}")
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            lon_MAE = test(model, test_loader, criterion, device, subtarget)


            """
            Lat
            """
            subtarget = 'lat'
            logger.info("Start loading data")
            images, labels, extra_params = load_train_dataset(subtarget)
            images, labels, extra_params = shuffle(images, labels, extra_params, random_state=777)
            train_len = int(0.9 * len(images))

            train_images, val_images = images[:train_len], images[train_len:]
            train_labels, val_labels = labels[:train_len], labels[train_len:]
            train_extra_params, val_extra_params = extra_params[:train_len], extra_params[train_len:]
            test_images, test_extra_params = load_test_dataset(subtarget)

            train_dataset = TrainDataset(train_images, train_labels, train_extra_params)
            val_dataset = TrainDataset(val_images, val_labels, val_extra_params)
            test_dataset = TestDataset(test_images, test_extra_params)
            
            logger.info("Start training CNN")
            model = CNN(model_type, use_extra_params).to(device)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Optimizer configuration
            base_params = [param for name, param in model.named_parameters() if param.requires_grad]
            optimizer = optim.Adam(base_params, lr=lr)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            min_MAE = 1000000

            for epoch in range(EPOCHS): #epoch
                train_loss = train(model, train_loader, criterion, optimizer, device)
                val_loss, MAE = validate(model, val_loader, criterion, device)
                scheduler.step(val_loss)

                if MAE < min_MAE:
                    min_MAE = MAE
                    torch.save(model.state_dict(), f'best_model({subtarget}).pth')
                logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | MAE: {MAE:.4f}")
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            lat_MAE = test(model, test_loader, criterion, device, subtarget)

            lon_df = pd.read_csv(f'Prediction(lon) - MAE = {lon_MAE}.csv')
            lon_df.rename(columns={'prediction': 'Lon prediction'}, inplace=True)
            lat_df = pd.read_csv(f'Prediction(lat) - MAE = {lat_MAE}.csv')
            lat_df.rename(columns={'prediction': 'Lat prediction'}, inplace=True)
            df = pd.merge(lon_df, lat_df, on='id')

            def compute_distance(row):
                point_a = (row['Lat'], row['Lon'])
                point_b = (row['Lat prediction'], row['Lon prediction'])
                return round(geodesic(point_a, point_b).kilometers, 3)
            df['distance'] = df.apply(compute_distance, axis=1)
            MAE = round(df['distance'].mean(), 3)
            df.to_csv(f'Prediction(loc) - MAE = {MAE}.csv', index=False)
            os.remove(f'Prediction(lon) - MAE = {lon_MAE}.csv')
            os.remove(f'Prediction(lat) - MAE = {lat_MAE}.csv')
        else:
            """
            load data
            """
            logger.info("Start loading data")
            images, labels, extra_params = load_train_dataset(target)
            images, labels, extra_params = shuffle(images, labels, extra_params, random_state=777)
            train_len = int(0.9 * len(images))

            train_images, val_images = images[:train_len], images[train_len:]
            train_labels, val_labels = labels[:train_len], labels[train_len:]
            train_extra_params, val_extra_params = extra_params[:train_len], extra_params[train_len:]
            test_images, test_extra_params = load_test_dataset(target)

            train_dataset = TrainDataset(train_images, train_labels, train_extra_params)
            val_dataset = TrainDataset(val_images, val_labels, val_extra_params)
            test_dataset = TestDataset(test_images, test_extra_params)
            
            """
            CNN - train and validate
            """
            logger.info("Start training CNN")
            model = CNN(model_type, use_extra_params).to(device)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Optimizer configuration
            base_params = [param for name, param in model.named_parameters() if param.requires_grad]
            optimizer = optim.Adam(base_params, lr=lr)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            min_MAE = 1000000

            for epoch in range(EPOCHS): #epoch
                train_loss = train(model, train_loader, criterion, optimizer, device)
                val_loss, MAE = validate(model, val_loader, criterion, device)
                scheduler.step(val_loss)

                if MAE < min_MAE:
                    min_MAE = MAE
                    torch.save(model.state_dict(), f'best_model({target}).pth')
                logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | MAE: {MAE:.4f}")

            """
            CNN - test
            """
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            MAE = test(model, test_loader, criterion, device, target)
        avg_mae += MAE
    avg_mae /= nRound
    logger.info(f"{nRound} rounds end, average MAE = {round(avg_mae, 3)}")

if __name__ == '__main__':
    main()
