import pickle
from sched import scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from sklearn.metrics import confusion_matrix
from utils.tools import EarlyStopping, adjust_learning_rate, save_to_csv, visual
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self._debug_print_shapes = True
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'val_class_accuracy': [],
            'test_class_accuracy': []
        }

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, external_scaler=None):
        try:
            data_set, data_loader = data_provider(self.args, flag, external_scaler=external_scaler)
        except TypeError:
            data_set, data_loader = data_provider(self.args, flag)
            if external_scaler is not None and hasattr(data_set, "sequences"):
                data_set.external_scaler = external_scaler
                data_set.scaler = external_scaler
                data_set.sequences = [data_set.scaler.transform(s) for s in data_set.sequences]
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_correct = 0
        total_samples = 0
        class_correct = [0] * self.args.num_classes
        class_total = [0] * self.args.num_classes
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, padding_masks) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)


                # encoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x,None,None,None)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x,None,None,None)
                    loss = criterion(outputs, batch_y)

                total_loss.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()

                for j in range(batch_y.size(0)):
                    label = batch_y[j]
                    class_total[label] += 1
                    if predicted[j] == label:
                        class_correct[label] += 1

        total_loss = np.average(total_loss)
        accuracy = 100 * total_correct / total_samples

        class_accuracy = []
        for i in range(self.args.num_classes):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
            else:
                class_acc = 0.0
            class_accuracy.append(class_acc)

        self.model.train()
        return total_loss, accuracy, class_accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train', external_scaler=None)
        scaler_save_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(scaler_save_dir, exist_ok=True)
        scaler_path = os.path.join(scaler_save_dir, 'scaler.pkl')
        train_scaler = getattr(train_data, "scaler", None)
        if train_scaler is not None:
            with open(scaler_path, 'wb') as f:
                pickle.dump(train_scaler, f)
            print(f"Saved train scaler to: {scaler_path}")
        else:
            print("Warning: train_scaler is None — dataset did not provide a scaler.")

        vali_data, vali_loader = self._get_data(flag='val', external_scaler=train_scaler)

        print("Train class counts:", np.bincount(train_data.labels.astype(int), minlength=self.args.num_classes))
        print("Val   class counts:", np.bincount(vali_data.labels.astype(int), minlength=self.args.num_classes))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        visual_path = os.path.join('./visual_results/', setting)
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.lradj == 'TST':
            from torch.optim import lr_scheduler
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_correct = 0
            train_total = 0

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, padding_masks) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, None, None)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, None, None, None)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

                if (i + 1) % 100 == 0:
                    train_accuracy = 100 * train_correct / train_total
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | acc: {3:.2f}%".format(
                        i + 1, epoch + 1, loss.item(), train_accuracy))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            train_accuracy = 100 * train_correct / train_total
            vali_loss, vali_accuracy, vali_class_accuracy = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train Acc: {3:.2f}% | Vali Loss: {4:.7f} Vali Acc: {5:.2f}%".format(
                    epoch + 1, train_steps, train_loss, train_accuracy, vali_loss, vali_accuracy))

            print("Vali Class Accuracies:")
            for i, acc in enumerate(vali_class_accuracy):
                print("  Class {0}: {1:.2f}%".format(i, acc))

            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_accuracy'].append(train_accuracy)
            self.train_history['val_loss'].append(vali_loss)
            self.train_history['val_accuracy'].append(vali_accuracy)
            self.train_history['val_class_accuracy'].append(vali_class_accuracy)

            if (epoch + 1) % 5 == 0 or epoch == self.args.train_epochs - 1:
                self._plot_training_curves(visual_path, epoch + 1)
                self._plot_class_accuracy(visual_path, epoch + 1)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                self._plot_training_curves(visual_path, epoch + 1, final=True)
                self._plot_class_accuracy(visual_path, epoch + 1, final=True)
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, None, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self._generate_training_report(visual_path)

        return self.model

    def _plot_training_curves(self, save_path, current_epoch, final=False):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(self.train_history['epoch'], self.train_history['train_loss'],
                 label='Train Loss', marker='o', linewidth=2)
        plt.plot(self.train_history['epoch'], self.train_history['val_loss'],
                 label='Val Loss', marker='s', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(self.train_history['epoch'], self.train_history['train_accuracy'],
                 label='Train Accuracy', marker='o', linewidth=2)
        plt.plot(self.train_history['epoch'], self.train_history['val_accuracy'],
                 label='Val Accuracy', marker='s', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        line1 = ax1.plot(self.train_history['epoch'], self.train_history['train_loss'],
                         'b-', label='Train Loss', linewidth=2)
        line2 = ax2.plot(self.train_history['epoch'], self.train_history['train_accuracy'],
                         'r-', label='Train Accuracy', linewidth=2)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax2.set_ylabel('Accuracy (%)', color='r')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        plt.title('Training Loss vs Accuracy')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.plot(self.train_history['epoch'], self.train_history['val_accuracy'],
                 label='Validation Accuracy', marker='s', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if final:
            filename = f'training_curves_final.png'
        else:
            filename = f'training_curves_epoch_{current_epoch}.png'

        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_class_accuracy(self, save_path, current_epoch, final=False):
        if len(self.train_history['val_class_accuracy']) == 0:
            return

        latest_val_acc = self.train_history['val_class_accuracy'][-1]

        plt.figure(figsize=(10, 4))
        val_matrix = np.array(latest_val_acc).reshape(1, -1)
        im = plt.imshow(val_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
        plt.title('Validation Class Accuracy (%)')
        plt.xlabel('Class')
        plt.ylabel('Epoch')
        plt.xticks(range(len(latest_val_acc)), [f'Class {i}' for i in range(len(latest_val_acc))])

        for i, acc in enumerate(latest_val_acc):
            plt.text(i, 0, f'{acc:.1f}%', ha='center', va='center',
                     fontweight='bold', color='white' if acc > 50 else 'black')

        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()

        if final:
            filename = f'class_accuracy_final.png'
        else:
            filename = f'class_accuracy_epoch_{current_epoch}.png'

        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.close()


    def _generate_training_report(self, save_path):
        if len(self.train_history['epoch']) == 0:
            return

        report_data = {
            'Epoch': self.train_history['epoch'],
            'Train_Loss': self.train_history['train_loss'],
            'Train_Accuracy': self.train_history['train_accuracy'],
            'Val_Loss': self.train_history['val_loss'],
            'Val_Accuracy': self.train_history['val_accuracy']
        }

        for class_idx in range(len(self.train_history['val_class_accuracy'][0])):
            report_data[f'Val_Class_{class_idx}_Accuracy'] = [
                epoch_acc[class_idx] for epoch_acc in self.train_history['val_class_accuracy']
            ]

        df_report = pd.DataFrame(report_data)
        csv_path = os.path.join(save_path, 'training_report.csv')
        df_report.to_csv(csv_path, index=False)

        final_stats = {
            'Best_Train_Accuracy': max(self.train_history['train_accuracy']),
            'Best_Val_Accuracy': max(self.train_history['val_accuracy']),
            'Final_Train_Accuracy': self.train_history['train_accuracy'][-1],
            'Final_Val_Accuracy': self.train_history['val_accuracy'][-1],
            'Total_Epochs': len(self.train_history['epoch']),
            'Best_Epoch_Val': self.train_history['epoch'][np.argmax(self.train_history['val_accuracy'])]
        }

        stats_path = os.path.join(save_path, 'training_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("=== TimeKAN Classification Training Report ===\n\n")
            for key, value in final_stats.items():
                if 'Accuracy' in key:
                    f.write(f"{key}: {value:.2f}%\n")
                else:
                    f.write(f"{key}: {value}\n")

            f.write(f"\nFinal Class Accuracies (Validation):\n")
            for i, acc in enumerate(self.train_history['val_class_accuracy'][-1]):
                f.write(f"  Class {i}: {acc:.2f}%\n")


    def test(self, setting, test=0):
        scaler_path = os.path.join(self.args.checkpoints, setting, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                train_scaler = pickle.load(f)
            print(f"✅ Loaded train scaler from: {scaler_path}")
        else:
            train_scaler = None
            print(f"⚠️ Warning: scaler not found at {scaler_path}. Test data will NOT be standardized consistently!")
        test_data, test_loader = self._get_data(flag='test', external_scaler=train_scaler)
        if test:
            print('loading model ...')
            best_model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        device = self.device

        all_preds = []
        all_trues = []
        all_probs = []

        total_correct = 0
        total_samples = 0
        class_correct = [0] * self.args.num_classes
        class_total = [0] * self.args.num_classes

        with torch.no_grad():
            for i, (batch_x, batch_y,padding_masks) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.long().to(device)

                # forward
                outputs = self.model(batch_x,None,None,None)
                probs = torch.softmax(outputs, dim=-1)

                preds = torch.argmax(probs, dim=1)

                # accumulate
                all_preds.append(preds.cpu().numpy())
                all_trues.append(batch_y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

                total_samples += batch_y.size(0)
                total_correct += (preds == batch_y).sum().item()

                for j in range(batch_y.size(0)):
                    label = batch_y[j]
                    class_total[label] += 1
                    if preds[j] == label:
                        class_correct[label] += 1


        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)
        all_probs = np.concatenate(all_probs)



        cm = confusion_matrix(all_trues, all_preds, labels=list(range(self.args.num_classes)))


        accuracy = 100.0 * (all_preds == all_trues).sum() / len(all_trues)
        print("Overall Accuracy: {:.2f}%".format(accuracy))


        class_accuracy = []
        for i in range(self.args.num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
            else:
                acc = 0.0
            class_accuracy.append(acc)
            print(f"  Class {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

        from utils.metrics import metric_classification
        precision, recall, f1, confusion_matrix_result = metric_classification(all_preds, all_trues)
        print("\nPrecision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}".format(precision, recall, f1))
        print("Confusion Matrix (metric_classification):\n", confusion_matrix_result)

        result_dir = f'./results/{setting}/'
        os.makedirs(result_dir, exist_ok=True)

        np.save(result_dir + 'pred.npy', all_preds)
        np.save(result_dir + 'true.npy', all_trues)
        np.save(result_dir + 'probs.npy', all_probs)
        np.save(result_dir + 'confusion_matrix.npy', cm)
        np.save(result_dir + 'class_accuracy.npy', np.array(class_accuracy))

        with open("result_classification.txt", 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n")
            f.write('Class Accuracies: ' + ', '.join(['{:.2f}%'.format(acc) for acc in class_accuracy]) + "\n\n")

        return accuracy