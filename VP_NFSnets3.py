# author Zihao Hu
# time 5/12/2020
# Modified for TensorFlow 2.x

# tensorflow_probabilityライブラリが必要です。
# pip install tensorflow-probability
import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp

# set random seed
np.random.seed(1234)
tf.random.set_seed(1234)

#############################################
###################VP NSFnet#################
#############################################

class VPNSFnet(tf.keras.Model):
    # Initialize the class
    def __init__(self, x0, y0, z0, t0, u0, v0, w0, xb, yb, zb, tb, ub, vb, wb, x, y, z, t, layers):
        super(VPNSFnet, self).__init__()

        Xb = np.concatenate([xb, yb, zb, tb], 1)

        self.lowb = tf.constant(Xb.min(0), dtype=tf.float32)
        self.upb = tf.constant(Xb.max(0), dtype=tf.float32)

        # 初期条件のデータをTensorに変換
        self.x0 = tf.constant(x0, dtype=tf.float32)
        self.y0 = tf.constant(y0, dtype=tf.float32)
        self.z0 = tf.constant(z0, dtype=tf.float32)
        self.t0 = tf.constant(t0, dtype=tf.float32)
        self.u0 = tf.constant(u0, dtype=tf.float32)
        self.v0 = tf.constant(v0, dtype=tf.float32)
        self.w0 = tf.constant(w0, dtype=tf.float32)

        # 境界条件のデータをTensorに変換
        self.xb = tf.constant(xb, dtype=tf.float32)
        self.yb = tf.constant(yb, dtype=tf.float32)
        self.zb = tf.constant(zb, dtype=tf.float32)
        self.tb = tf.constant(tb, dtype=tf.float32)
        self.ub = tf.constant(ub, dtype=tf.float32)
        self.vb = tf.constant(vb, dtype=tf.float32)
        self.wb = tf.constant(wb, dtype=tf.float32)

        # 物理法則(f)のデータをTensorに変換
        self.x = tf.constant(x, dtype=tf.float32)
        self.y = tf.constant(y, dtype=tf.float32)
        self.z = tf.constant(z, dtype=tf.float32)
        self.t = tf.constant(t, dtype=tf.float32)
        
        # ニューラルネットワークの構築 (Keras Sequential APIを使用)
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        # Normalization layer
        self.net.add(tf.keras.layers.Lambda(lambda X: 2.0 * (X - self.lowb) / (self.upb - self.lowb) - 1.0))
        for i in range(len(layers) - 2):
            self.net.add(tf.keras.layers.Dense(layers[i+1],
                                               activation='tanh',
                                               kernel_initializer='glorot_normal')) # Xavier Initializer
        self.net.add(tf.keras.layers.Dense(layers[-1])) # 最終層は活性化関数なし

    # 順伝播
    def call(self, x, y, z, t):
        X = tf.concat([x, y, z, t], 1)
        u_v_w_p = self.net(X)
        u = u_v_w_p[:, 0:1]
        v = u_v_w_p[:, 1:2]
        w = u_v_w_p[:, 2:3]
        p = u_v_w_p[:, 3:4]
        return u, v, w, p

    # 物理法則に関する残差(f)を計算
    def net_f_NS(self, x, y, z, t):
        Re = 1.0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z, t])
            u, v, w, p = self.call(x, y, z, t)

            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_z = tape.gradient(u, z)
            u_t = tape.gradient(u, t)
            
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            v_z = tape.gradient(v, z)
            v_t = tape.gradient(v, t)

            w_x = tape.gradient(w, x)
            w_y = tape.gradient(w, y)
            w_z = tape.gradient(w, z)
            w_t = tape.gradient(w, t)

            p_x = tape.gradient(p, x)
            p_y = tape.gradient(p, y)
            p_z = tape.gradient(p, z)
        
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        u_zz = tape.gradient(u_z, z)

        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)
        v_zz = tape.gradient(v_z, z)

        w_xx = tape.gradient(w_x, x)
        w_yy = tape.gradient(w_y, y)
        w_zz = tape.gradient(w_z, z)
        
        del tape

        f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1/Re) * (u_xx + u_yy + u_zz)
        f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1/Re) * (v_xx + v_yy + v_zz)
        f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1/Re) * (w_xx + w_yy + w_zz)
        f_e = u_x + v_y + w_z
        
        return u, v, w, p, f_u, f_v, f_w, f_e

    # 損失関数
    def compute_loss(self):
        u_ini_pred, v_ini_pred, w_ini_pred, p_ini_pred = self.call(self.x0, self.y0, self.z0, self.t0)
        u_b_pred, v_b_pred, w_b_pred, p_b_pred = self.call(self.xb, self.yb, self.zb, self.tb)
        _, _, _, _, f_u_pred, f_v_pred, f_w_pred, f_e_pred = self.net_f_NS(self.x, self.y, self.z, self.t)
        
        alpha = 100.0
        beta = 100.0

        loss = alpha * tf.reduce_mean(tf.square(self.u0 - u_ini_pred)) + \
               alpha * tf.reduce_mean(tf.square(self.v0 - v_ini_pred)) + \
               alpha * tf.reduce_mean(tf.square(self.w0 - w_ini_pred)) + \
               beta * tf.reduce_mean(tf.square(self.ub - u_b_pred)) + \
               beta * tf.reduce_mean(tf.square(self.vb - v_b_pred)) + \
               beta * tf.reduce_mean(tf.square(self.wb - w_b_pred)) + \
               tf.reduce_mean(tf.square(f_u_pred)) + \
               tf.reduce_mean(tf.square(f_v_pred)) + \
               tf.reduce_mean(tf.square(f_w_pred)) + \
               tf.reduce_mean(tf.square(f_e_pred))
        return loss

    # Adamオプティマイザ用の学習ステップ
    @tf.function
    def train_step_adam(self, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss()
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def Adam_train(self, nIter=5000, learning_rate=1e-3):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        start_time = time.time()
        for it in range(nIter):
            loss_value = self.train_step_adam(optimizer)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % (it, loss_value, elapsed))
                start_time = time.time()
    
    # BFGSオプティマイザ用の損失と勾配を計算する関数
    def loss_and_grads_for_bfgs(self, weights):
        with tf.GradientTape() as tape:
            self.set_weights(weights)
            loss = self.compute_loss()
        grads = tape.gradient(loss, self.trainable_variables)
        return loss, tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
    
    def callback(self, loss):
        print('Loss: %.3e' % loss)

    def BFGS_train(self):
        # TensorFlow ProbabilityのLBFGSオプティマイザを使用
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=self.loss_and_grads_for_bfgs,
            initial_position=tf.concat([tf.reshape(v, [-1]) for v in self.trainable_variables], axis=0),
            max_iterations=50000,
            num_correction_pairs=50,
            f_relative_tolerance=1.0 * np.finfo(float).eps)
        
        # 最適化された重みをモデルに設定
        self.set_weights(results.position)
        self.callback(results.objective_value)
    
    def predict(self, x_star, y_star, z_star, t_star):
        u_star, v_star, w_star, p_star = self.call(
            tf.constant(x_star, dtype=tf.float32),
            tf.constant(y_star, dtype=tf.float32),
            tf.constant(z_star, dtype=tf.float32),
            tf.constant(t_star, dtype=tf.float32)
        )
        return u_star.numpy(), v_star.numpy(), w_star.numpy(), p_star.numpy()


if __name__ == "__main__":
    N_train = 10000
    layers = [4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 4]

    # Load Data
    def data_generate(x, y, z, t):
        a, d = 1, 1
        u = - a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(- d * d * t)
        v = - a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(- d * d * t)
        w = - a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(- d * d * t)
        p = - 0.5 * a * a * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
                             2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
                             2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
                             2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) * np.exp(
            -2 * d * d * t)
        return u, v, w, p

    # --- データ準備 (元のコードのバグを修正) ---
    x1 = np.linspace(-1, 1, 31)
    y1 = np.linspace(-1, 1, 31)
    z1 = np.linspace(-1, 1, 31)
    t1 = np.linspace(0, 1, 11)
    b0 = np.array([-1] * 900)
    b1 = np.array([1] * 900)

    xt = np.tile(x1[0:30], 30)
    yt = np.tile(y1[0:30], 30)
    zt = np.tile(z1[0:30], 30)
    xt1 = np.tile(x1[1:31], 30)
    yt1 = np.tile(y1[1:31], 30)
    zt1 = np.tile(z1[1:31], 30)

    xr = x1[0:30].repeat(30)
    yr = y1[0:30].repeat(30)
    zr = z1[0:30].repeat(30)
    xr1 = x1[1:31].repeat(30)
    yr1 = y1[1:31].repeat(30)
    zr1 = z1[1:31].repeat(30)

    train1x = np.concatenate([b1, b0, xt1, xt, xt1, xt], 0).repeat(t1.shape[0])
    train1y = np.concatenate([yt, yt1, b1, b0, yr1, yr], 0).repeat(t1.shape[0])
    train1z = np.concatenate([zr, zr1, zr, zr1, b1, b0], 0).repeat(t1.shape[0])
    train1t = np.tile(t1, 5400)
    
    train1ub, train1vb, train1wb, train1pb = data_generate(train1x, train1y, train1z, train1t)

    # reshapeのバグを修正
    xb_train = train1x.reshape(-1, 1)
    yb_train = train1y.reshape(-1, 1)
    zb_train = train1z.reshape(-1, 1)
    tb_train = train1t.reshape(-1, 1)
    ub_train = train1ub.reshape(-1, 1)
    vb_train = train1vb.reshape(-1, 1)
    wb_train = train1wb.reshape(-1, 1)

    x_0 = np.tile(x1, 31 * 31)
    y_0 = np.tile(y1.repeat(31), 31)
    z_0 = z1.repeat(31 * 31)
    t_0 = np.array([0] * x_0.shape[0])

    u_0, v_0, w_0, p_0 = data_generate(x_0, y_0, z_0, t_0)

    u0_train = u_0.reshape(-1, 1)
    v0_train = v_0.reshape(-1, 1)
    w0_train = w_0.reshape(-1, 1)
    x0_train = x_0.reshape(-1, 1)
    y0_train = y_0.reshape(-1, 1)
    z0_train = z_0.reshape(-1, 1)
    t0_train = t_0.reshape(-1, 1)

    # 物理法則(f)の訓練データ
    xx = np.random.uniform(-1, 1, size=10000)
    yy = np.random.uniform(-1, 1, size=10000)
    zz = np.random.uniform(-1, 1, size=10000)
    tt = np.random.uniform(0, 1, size=10000)

    x_train = xx.reshape(-1, 1)
    y_train = yy.reshape(-1, 1)
    z_train = zz.reshape(-1, 1)
    t_train = tt.reshape(-1, 1)

    model = VPNSFnet(x0_train, y0_train, z0_train, t0_train,
                     u0_train, v0_train, w0_train,
                     xb_train, yb_train, zb_train, tb_train,
                     ub_train, vb_train, wb_train,
                     x_train, y_train, z_train, t_train, layers)
    
    # 学習の実行
    model.Adam_train(5000, 1e-3)
    model.Adam_train(5000, 1e-4)
    model.Adam_train(50000, 1e-5)
    model.Adam_train(50000, 1e-6)
    model.BFGS_train()
    
    # テストデータ
    x_star = (np.random.rand(1000, 1) - 0.5) * 2
    y_star = (np.random.rand(1000, 1) - 0.5) * 2
    z_star = (np.random.rand(1000, 1) - 0.5) * 2
    t_star = np.random.rand(1000, 1)

    u_star, v_star, w_star, p_star = data_generate(x_star, y_star, z_star, t_star)

    # 予測
    u_pred, v_pred, w_pred, p_pred = model.predict(x_star, y_star, z_star, t_star)

    # 誤差の計算
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    # 3つ目のprintのtypoを修正
    print('Error w: %e' % error_w)
    print('Error p: %e' % error_p)