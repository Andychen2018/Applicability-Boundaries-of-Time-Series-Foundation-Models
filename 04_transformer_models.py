"""
04_Transformer模型
使用Transformer架构进行时间序列分类
支持三种分类方式：ShengYing、ZhenDong、Fusion
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_utils import MotorDataLoader
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=6, num_classes=3, max_len=512):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(x)
        return x

class TransformerTrainer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.results = {}
        
    def train_model(self, model, train_loader, val_loader, epochs=50, lr=0.001):
        """训练Transformer模型"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        model.to(self.device)
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 10
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            scheduler.step(val_loss)
            
            if epoch % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        return train_losses, val_losses, val_accuracies, best_val_acc
    
    def evaluate_model(self, model, test_loader):
        """评估模型"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        return all_labels, all_predictions, accuracy
    
    def train_and_evaluate(self, mode='shengying'):
        """训练和评估Transformer模型"""
        print(f"\n=== Training Transformer models for {mode} mode ===")
        
        # 加载数据
        loader = MotorDataLoader()
        X_raw, y = loader.load_data(mode=mode)
        
        # 下采样以减少计算量
        downsample_factor = 128  # 从65536降到512
        X_downsampled = X_raw[:, ::downsample_factor]
        
        print(f"Original shape: {X_raw.shape}, Downsampled shape: {X_downsampled.shape}")
        
        # 分割数据
        X_train, X_test, y_train, y_test = loader.split_data(X_downsampled, y)
        
        # 标准化数据
        X_train_scaled, X_test_scaled, scaler = loader.normalize_data(X_train, X_test)
        
        # 重塑数据为Transformer输入格式 (batch_size, seq_len, input_dim)
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        # 创建数据集和数据加载器
        train_dataset = TimeSeriesDataset(X_train_reshaped, y_train)
        test_dataset = TimeSeriesDataset(X_test_reshaped, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 创建验证集
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        # 定义不同的Transformer配置
        configs = {
            'Small': {'d_model': 64, 'nhead': 4, 'num_layers': 3},
            'Medium': {'d_model': 128, 'nhead': 8, 'num_layers': 6},
            'Large': {'d_model': 256, 'nhead': 8, 'num_layers': 8}
        }
        
        mode_results = {}
        
        for config_name, config in configs.items():
            print(f"\nTraining Transformer-{config_name}...")
            
            # 创建模型
            model = TransformerClassifier(
                input_dim=1,
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                num_classes=len(np.unique(y)),
                max_len=X_train_reshaped.shape[1]
            )
            
            # 训练模型
            train_losses, val_losses, val_accuracies, best_val_acc = self.train_model(
                model, train_loader, val_loader, epochs=50, lr=0.001
            )
            
            # 评估模型
            y_true, y_pred, test_accuracy = self.evaluate_model(model, test_loader)
            
            # 保存结果
            mode_results[config_name] = {
                'test_accuracy': test_accuracy,
                'best_val_accuracy': best_val_acc,
                'y_true': y_true,
                'y_pred': y_pred,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            
            print(f"Transformer-{config_name} - Test Accuracy: {test_accuracy:.4f}, Best Val Accuracy: {best_val_acc:.4f}")
            
            # 保存模型
            model_path = os.path.join(self.output_dir, 'table', f'04_{mode}_transformer_{config_name.lower()}_model.pth')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'config': config,
                'test_accuracy': test_accuracy
            }, model_path)
        
        self.results[mode] = mode_results
        
        # 生成报告和可视化
        self._generate_reports(mode)
        self._plot_results(mode)
        
        return mode_results
    
    def _generate_reports(self, mode):
        """生成结果报告"""
        results_df = []
        
        for model_name, result in self.results[mode].items():
            results_df.append({
                'Model': f'Transformer-{model_name}',
                'Mode': mode,
                'Test_Accuracy': result['test_accuracy'],
                'Best_Val_Accuracy': result['best_val_accuracy']
            })
        
        df = pd.DataFrame(results_df)
        
        # 保存结果表格
        table_path = os.path.join(self.output_dir, 'table', f'04_{mode}_transformer_results.csv')
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        df.to_csv(table_path, index=False)
        
        print(f"\nResults saved to {table_path}")
        print(df.to_string(index=False))
    
    def _plot_results(self, mode):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 准确率对比
        model_names = [f'Transformer-{name}' for name in self.results[mode].keys()]
        test_accuracies = [self.results[mode][name]['test_accuracy'] for name in self.results[mode].keys()]
        val_accuracies = [self.results[mode][name]['best_val_accuracy'] for name in self.results[mode].keys()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, test_accuracies, width, label='Test Accuracy', alpha=0.7)
        axes[0, 0].bar(x + width/2, val_accuracies, width, label='Best Val Accuracy', alpha=0.7)
        axes[0, 0].set_title(f'Transformer Model Accuracy Comparison - {mode}')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        
        # 混淆矩阵 (使用最佳模型)
        best_model = max(self.results[mode].keys(), key=lambda x: self.results[mode][x]['test_accuracy'])
        y_true = self.results[mode][best_model]['y_true']
        y_pred = self.results[mode][best_model]['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
        axes[0, 1].set_title(f'Confusion Matrix - Transformer-{best_model} ({mode})')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 训练曲线
        train_losses = self.results[mode][best_model]['train_losses']
        val_losses = self.results[mode][best_model]['val_losses']
        
        axes[1, 0].plot(train_losses, label='Train Loss')
        axes[1, 0].plot(val_losses, label='Val Loss')
        axes[1, 0].set_title(f'Training Curves - Transformer-{best_model} ({mode})')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        
        # 性能总结
        axes[1, 1].text(0.1, 0.8, f'Best Model: Transformer-{best_model}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Test Accuracy: {self.results[mode][best_model]["test_accuracy"]:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Val Accuracy: {self.results[mode][best_model]["best_val_accuracy"]:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title(f'Performance Summary - {mode}')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        img_path = os.path.join(self.output_dir, 'images', f'04_{mode}_transformer_results.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results plot saved to {img_path}")

def main():
    """主函数"""
    trainer = TransformerTrainer()
    
    # 对三种模式分别进行实验
    modes = ['shengying', 'zhendong', 'fusion']
    
    for mode in modes:
        try:
            trainer.train_and_evaluate(mode=mode)
        except Exception as e:
            print(f"Error in {mode} mode: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n=== All Transformer experiments completed ===")

if __name__ == "__main__":
    main()
