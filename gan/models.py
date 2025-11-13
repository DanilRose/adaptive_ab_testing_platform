import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.special import softmax
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims, dropout_rate):
        super(Generator, self).__init__()
        layers = []
        input_dim = latent_dim
        
        layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        ])
        input_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        layers.extend([
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Tanh()
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.network(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate, leaky_slope):
        super(Discriminator, self).__init__()
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(leaky_slope),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        layers.extend([
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class WGAN_GP_Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims):
        super(WGAN_GP_Generator, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        self.res_blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        
        current_dim = hidden_dims[0]
        for i in range(len(hidden_dims) - 1):
            res_block = nn.Sequential(
                nn.Linear(current_dim, hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dims[i+1], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1])
            )
            self.res_blocks.append(res_block)
            
            if current_dim != hidden_dims[i+1]:
                shortcut = nn.Linear(current_dim, hidden_dims[i+1])
                self.shortcuts.append(shortcut)
            else:
                self.shortcuts.append(nn.Identity())
            
            current_dim = hidden_dims[i+1]
        
        self.output = nn.Linear(current_dim, output_dim)
        self.tanh = nn.Tanh()
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = nn.LeakyReLU(0.2)(x)
        
        for i, (res_block, shortcut) in enumerate(zip(self.res_blocks, self.shortcuts)):
            residual = shortcut(x)  
            x = res_block(x) + residual
            x = nn.LeakyReLU(0.2)(x)
        
        return self.tanh(self.output(x))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class WGAN_GP_Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(WGAN_GP_Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        
        self.res_blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        
        current_dim = hidden_dims[0]
        for i in range(len(hidden_dims) - 1):
            res_block = nn.Sequential(
                nn.Linear(current_dim, hidden_dims[i+1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Linear(hidden_dims[i+1], hidden_dims[i+1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            )
            self.res_blocks.append(res_block)
            
            if current_dim != hidden_dims[i+1]:
                shortcut = nn.Linear(current_dim, hidden_dims[i+1])
                self.shortcuts.append(shortcut)
            else:
                self.shortcuts.append(nn.Identity())
            
            current_dim = hidden_dims[i+1]
        
        self.output = nn.Linear(current_dim, 1)
        
        self._initialize_weights()
    
    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        
        for res_block, shortcut in zip(self.res_blocks, self.shortcuts):
            residual = shortcut(x)  
            x = res_block(x) + residual
        
        return self.output(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class UserDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class GAN:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.g_losses = []
        self.d_losses = []
        self.gradient_penalties = []
        self.wasserstein_distances = []
        self.scalers = {}
        self.encoders = {}
        self.feature_info = {}
        self.imputers = {}
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def prepare_data(self, real_data):
        self.real_data = real_data.copy()
        processed_data = pd.DataFrame()
        
        numerical_features = real_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = real_data.select_dtypes(include=['object']).columns.tolist()
        boolean_features = real_data.select_dtypes(include=['bool']).columns.tolist()
        
        for feature in numerical_features:
            if real_data[feature].isnull().any():
                imputer = SimpleImputer(strategy='median')
                real_data[feature] = imputer.fit_transform(real_data[[feature]])
                self.imputers[feature] = imputer
            
            scaler = StandardScaler()
            processed_data[feature] = scaler.fit_transform(real_data[[feature]].values.reshape(-1, 1)).flatten()
            self.scalers[feature] = scaler
            self.feature_info[feature] = {'type': 'numerical', 'scaler': scaler}
        
        for feature in categorical_features:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(real_data[feature].astype(str))
            processed_data[feature] = encoded
            self.encoders[feature] = encoder
            self.feature_info[feature] = {
                'type': 'categorical',
                'encoder': encoder,
                'classes': encoder.classes_.tolist()
            }
        
        for feature in boolean_features:
            processed_data[feature] = real_data[feature].astype(float)
            self.feature_info[feature] = {'type': 'boolean'}
        
        self.processed_columns = processed_data.columns.tolist()
        self.input_dim = len(self.processed_columns)
        
        if self.config.USE_WGAN_GP:
            self.generator = WGAN_GP_Generator(
                self.config.LATENT_DIM,
                self.input_dim,
                self.config.GENERATOR_LAYERS
            ).to(self.device)
            self.discriminator = WGAN_GP_Discriminator(
                self.input_dim,
                self.config.DISCRIMINATOR_LAYERS
            ).to(self.device)
        else:
            self.generator = Generator(
                self.config.LATENT_DIM,
                self.input_dim,
                self.config.GENERATOR_LAYERS,
                self.config.DROPOUT_RATE
            ).to(self.device)
            self.discriminator = Discriminator(
                self.input_dim,
                self.config.DISCRIMINATOR_LAYERS,
                self.config.DROPOUT_RATE,
                self.config.LEAKY_RELU_SLOPE
            ).to(self.device)
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.config.LEARNING_RATE, betas=(0.0, 0.9))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.config.LEARNING_RATE, betas=(0.0, 0.9))
        
        self.criterion = nn.BCELoss()
        
        print(f"Размерность: {self.input_dim} (без one-hot)")
        print(f"Параметры генератора: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Параметры дискриминатора: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        return torch.FloatTensor(processed_data.values).to(self.device)
    
    def train_epoch_standard(self, dataloader, epoch):
        for i, real_batch in enumerate(dataloader):
            batch_size = real_batch.size(0)
            
            real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9  
            fake_labels = torch.zeros(batch_size, 1).to(self.device) * 0.1 
            
            # Обучение дискриминатора
            self.discriminator.zero_grad()
            
            real_output = self.discriminator(real_batch)
            d_loss_real = self.criterion(real_output, real_labels)
            
            z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
            fake_batch = self.generator(z)
            fake_output = self.discriminator(fake_batch.detach())
            d_loss_fake = self.criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optimizer_D.step()
            
            # Обучение генератора
            self.generator.zero_grad()
            
            gen_output = self.discriminator(fake_batch)
            g_loss = self.criterion(gen_output, real_labels)
            g_loss.backward()
            self.optimizer_G.step()
            
            self.g_losses.append(g_loss.item())
            self.d_losses.append(d_loss.item())
    
    def train_epoch_wgan_gp(self, dataloader, epoch):
        for i, real_batch in enumerate(dataloader):
            batch_size = real_batch.size(0)
            
            for _ in range(self.config.N_CRITIC):
                self.discriminator.zero_grad()
                
                z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
                fake_batch = self.generator(z)
                
                real_output = self.discriminator(real_batch)
                fake_output = self.discriminator(fake_batch)
                
                gradient_penalty = self.compute_gradient_penalty(real_batch.data, fake_batch.data)
                d_loss = -torch.mean(real_output) + torch.mean(fake_output) + self.config.LAMBDA_GP * gradient_penalty
                
                d_loss.backward()
                self.optimizer_D.step()
                
                self.gradient_penalties.append(gradient_penalty.item())
            
            self.generator.zero_grad()
            fake_batch = self.generator(z)
            fake_output = self.discriminator(fake_batch)
            g_loss = -torch.mean(fake_output)
            
            g_loss.backward()
            self.optimizer_G.step()
            
            self.g_losses.append(g_loss.item())
            self.d_losses.append(d_loss.item())
            
            wasserstein_distance = torch.mean(real_output) - torch.mean(fake_output)
            self.wasserstein_distances.append(wasserstein_distance.item())
    
    def train(self, real_data, epochs=None):
        epochs = epochs or self.config.EPOCHS
        
        X_train = self.prepare_data(real_data)
        dataset = UserDataset(X_train.cpu().numpy())
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        print(f"Начало обучения на {epochs} эпох")
        print(f"Режим: {'WGAN-GP' if self.config.USE_WGAN_GP else 'Standard GAN'}")
        
        best_fid = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            if self.config.USE_WGAN_GP:
                self.train_epoch_wgan_gp(dataloader, epoch)
            else:
                self.train_epoch_standard(dataloader, epoch)
            
            # Логирование
            if epoch % self.config.LOG_INTERVAL == 0:
                current_g_loss = self.g_losses[-1] if self.g_losses else 0
                current_d_loss = self.d_losses[-1] if self.d_losses else 0
                
                if self.config.USE_WGAN_GP:
                    wasserstein = abs(self.wasserstein_distances[-1]) if self.wasserstein_distances else 0
                    gp = self.gradient_penalties[-1] if self.gradient_penalties else 0
                    print(f"Epoch [{epoch}/{epochs}] G: {current_g_loss:.4f} D: {current_d_loss:.4f} "
                          f"W: {wasserstein:.4f} GP: {gp:.4f}")
                else:
                    print(f"Epoch [{epoch}/{epochs}] G: {current_g_loss:.4f} D: {current_d_loss:.4f}")
            
            # мониторинг
            if epoch % self.config.VALIDATION_INTERVAL == 0 and epoch > 0:
                fid_score = self._validate_training(real_data, epoch)
                
                if fid_score < best_fid:
                    best_fid = fid_score
                    patience_counter = 0
                    self._save_checkpoint(f"best_fid_{fid_score:.1f}")
                    print(f"✅ Новый лучший FID: {fid_score:.1f}")
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        break
            
            if epoch % self.config.CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(epoch)
            
            if epoch % self.config.SAMPLE_INTERVAL == 0:
                self._generate_samples(epoch, self.config.SAMPLE_SIZE)
        
        print("Обучение завершено!")
        self._save_checkpoint("final")
    
    def _validate_training(self, real_data, epoch):
        try:
            synthetic_val = self.generate(5000)
            
            from scripts.evaluator import GANEvaluator
            evaluator = GANEvaluator(real_data.sample(5000), synthetic_val, scalers=self.scalers)
            fid_score = evaluator.calculate_fid_score()
            
            numerical_features = real_data.select_dtypes(include=[np.number]).columns
            numerical_features = [f for f in numerical_features if f in synthetic_val.columns]
            stats_ok = 0
            
            for feature in numerical_features[:5]:
                real_mean = real_data[feature].mean()
                synth_mean = synthetic_val[feature].mean()
                mean_diff = abs(real_mean - synth_mean) / real_mean
                
                if mean_diff < 0.2:  
                    stats_ok += 1
            
            print(f" Validation Epoch {epoch}: FID={fid_score:.1f}, Stats: {stats_ok}/5 OK")
            return fid_score
        
        except Exception as e:
            print(f" Validation error: {e}")
            return float('inf')
    
    def _save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'feature_info': self.feature_info,
            'processed_columns': self.processed_columns,
            'scalers': self.scalers
        }, f'gan_checkpoint_epoch_{epoch}.pth')
    
    def _generate_samples(self, epoch, n_samples):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.LATENT_DIM).to(self.device)
            synthetic_data = self.generator(z).cpu().numpy()
            synthetic_df = self._postprocess(synthetic_data)
            synthetic_df.to_csv(f'samples_epoch_{epoch}.csv', index=False)
        self.generator.train()
    
    def _postprocess(self, synthetic_data):
        synthetic_df = pd.DataFrame(synthetic_data, columns=self.processed_columns)
        result_df = pd.DataFrame()
        
        device_classes = self.feature_info['device']['classes']
        device_indices = np.random.randint(0, len(device_classes), len(synthetic_df))
        result_df['device'] = [device_classes[idx] for idx in device_indices]
        
        for i, device in enumerate(result_df['device']):
            if device == 'Mobile':
                os_options = ['iOS', 'Android']
            elif device == 'Desktop':
                os_options = ['Windows', 'macOS'] 
            else:  # Tablet
                os_options = ['iOS', 'Android']
            
            result_df.at[i, 'os'] = np.random.choice(os_options)
        
        for i, (device, os) in enumerate(zip(result_df['device'], result_df['os'])):
            if device == 'Mobile':
                if os == 'iOS':
                    browser_options = ['Safari Mobile', 'Chrome Mobile']
                else:  
                    browser_options = ['Chrome Mobile', 'Firefox Mobile', 'Samsung Internet']
            elif device == 'Desktop':
                if os == 'Windows':
                    browser_options = ['Chrome', 'Firefox', 'Edge', 'Safari']
                else: 
                    browser_options = ['Safari', 'Chrome', 'Firefox']
            else:  
                if os == 'iOS':
                    browser_options = ['Safari Mobile', 'Chrome Mobile']
                else:  
                    browser_options = ['Chrome Mobile', 'Firefox Mobile', 'Samsung Internet']
            
            result_df.at[i, 'browser'] = np.random.choice(browser_options)
        

        other_categorical = [f for f in self.feature_info.keys() 
                        if self.feature_info[f]['type'] == 'categorical'
                        and f not in ['device', 'os', 'browser']]
        
        for feature in other_categorical:
            classes = self.feature_info[feature]['classes']
            values = np.random.randint(0, len(classes), len(synthetic_df))
            result_df[feature] = [classes[idx] for idx in values]
        
        numerical_features = [f for f in self.feature_info.keys() 
                            if self.feature_info[f]['type'] == 'numerical']
        
        for feature in numerical_features:
            scaler = self.feature_info[feature]['scaler']
            values = scaler.inverse_transform(synthetic_df[feature].values.reshape(-1, 1)).flatten()
            
            if feature in ['age', 'previous_purchases', 'pages_per_session', 'visits_per_week']:
                values = np.clip(values.round().astype(int), 0, None)
            if feature == 'age':
                values = np.clip(values, 18, 70)
            if feature == 'total_spent':
                values = np.clip(values, 0, 300000)
            if feature == 'income':
                values = np.clip(values, 20000, 200000)
            
            result_df[feature] = values
        
        boolean_features = [f for f in self.feature_info.keys() 
                        if self.feature_info[f]['type'] == 'boolean']
        
        for feature in boolean_features:
            result_df[feature] = (synthetic_df[feature] > 0.5)
        
        result_df['user_id'] = np.arange(len(result_df))
        
        cols = ['user_id'] + [col for col in result_df.columns if col != 'user_id']
        result_df = result_df[cols]
        
        return result_df
    
    def generate(self, n_samples=10000):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.LATENT_DIM).to(self.device)
            synthetic_data = self.generator(z).cpu().numpy()
            return self._postprocess(synthetic_data)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        self.feature_info = checkpoint['feature_info']
        self.processed_columns = checkpoint['processed_columns']
        self.scalers = checkpoint.get('scalers', {})
        print(f"Загружена модель из эпохи {checkpoint['epoch']}")