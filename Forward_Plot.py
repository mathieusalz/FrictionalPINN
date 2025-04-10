import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
from numpy import format_float_scientific as ffs
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from Forward_Parameter import t1, t2, vpl, statepl, dc, v_ini, state_ini

device = 'cpu'

class Result:
    
    def __init__(self, NN, loss_list, name):
        self.NN = NN
        self.loss_list = loss_list
        self.name = name
        
    def Input(self):
        PINN = self.NN
        output_span = 10 * 3600

        t4outp = PINN.time
        
        upred = PINN.forward(t4outp)
        upred = upred.cpu().detach().numpy()
        Nt = len(t4outp)
        upred = np.reshape(upred,(Nt,2),order='F')

        t4outp = t4outp
        p_out = upred[:, 0]
        q_out = upred[:, 1]
        v_out = vpl * np.exp(p_out)
        state_out = statepl * np.exp(q_out)

        return t4outp, v_out, state_out
 
    def plot_vtheta(self, save = False):
        PINN = self.NN
        
        t4outp, v_out, state_out = self.Input()

        t_toPlot = t4outp * PINN.nonNormalizer / (3600 * 24)

        fig = plt.figure(figsize = (10, 10 / 3), dpi = 300)
        fig.patch.set_facecolor('white')

        ax = fig.add_subplot(121)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y",useMathText=True)
        ax.plot(t_toPlot, v_out, label = "PINN", color = "tab:red", linewidth = 3)
        ax.set_yscale('log')
        ax.set_xlabel(r"$t$ [days]")
        ax.set_ylabel(r"$v$ [m/s]")
        ax.legend(fontsize = "large")
        ax.set_xticks(range(0, 801, 200))


        ax2 = fig.add_subplot(122)
        ax2.ticklabel_format(style="sci", scilimits=(0,0), axis="y",useMathText=True)
        ax2.plot(t_toPlot, state_out, label = "PINN", color = "tab:red", linewidth = 3)
        ax2.set_yscale('log')
        ax2.set_xlabel(r"$t$ [days]")
        ax2.set_ylabel(r"$\theta$ [s]")
        ax2.set_xticks(range(0, 801, 200))


        figname = self.name + "_vtheta.png"
        plt.tight_layout()

        if(save):
            fig.savefig(figname ,bbox_inches="tight", pad_inches=0.05)
        else:
            plt.show()
           
    def plot_loss(self, save = False):
        loss = self.loss_list["total"]
        loss_ini = self.loss_list["ini"]
        loss_f = self.loss_list["ode"]
        
        print("L    : ", ffs(loss[-1]  , 3, 2))
        print("Lode : ", ffs(loss_f[-1], 3, 2))
        print("Lini : ", ffs(loss_ini[-1], 3,2))
        print("Iteration : ", len(loss))

        fig = plt.figure(figsize = (5, 10/3), dpi = 300)
        fig.patch.set_facecolor('white')

        ax = fig.add_subplot(1, 1, 1)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y",useMathText=True)
        
        xiter = np.arange(0, len(loss_ini))

        ax.plot(xiter, loss_ini, label = r"$L_{ini}$", linewidth = 3, color = "tab:blue")
        ax.plot(xiter, loss_f, label = r"$ L_{ode}$", linewidth = 3, color = "tab:red")
        ax.set_yscale("log")

        ax.set_xlabel("Iteration", fontsize = 14)
        ax.set_ylabel("Loss", fontsize = 14)
        ax.legend(fontsize = "large") #loc = 'upper right'

        ax.set_ylim(1e-8,1e2)

        figname = self.name + '_loss.png'
        if(save):
            fig.savefig(figname ,bbox_inches="tight", pad_inches=0.05)
        else:
            plt.show()

        return
    
    def animation_plot(self, loss_list, t_test):
        
        plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\ffmpeg.exe'
        # Create and save animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


        def update(frame):
            ax1.clear()
            ax2.clear()
            ax1.plot(t_test, loss_list['p'][frame], color='green')
            ax2.plot(t_test, loss_list['q'][frame], color='blue')
            ax1.set_title('P Prediction')
            ax2.set_title('Q Prediction')

        ani = FuncAnimation(fig, update, frames=len(loss_list['p']), repeat=False)
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("training_animation.mp4", writer=writer)


    def animation_plotSpecial(self):

        PINN = self.NN

        iter_skip = int(self.loss_list['iter'][1] - self.loss_list['iter'][0])

        t_test = self.NN.time * PINN.nonNormalizer / (3600 * 24)

        # Create figure with GridSpec
        fig = plt.figure(figsize=(12, 15))
        gs = GridSpec(2, 2, height_ratios=[1, 1])  # 3 rows, 2 columns

        # Define subplots
        axes = [
            fig.add_subplot(gs[0, 0]),  # Top-left
            fig.add_subplot(gs[0, 1]),  # Top-right
        ]
        ax_bottom = fig.add_subplot(gs[1, :])  # Bottom row spanning both columns


        # **Static axis settings**
        for ax in axes:
            ax.set_xlabel('Time [days]')
            ax.set_yscale('log')

        axes[0].set_title('V Prediction')
        axes[1].set_title(r'$\theta$ Prediction')

        axes[0].scatter(t_test[0], vpl * np.exp(self.NN.u_ini[0]), color='k', alpha=0.5, s=50)
        axes[1].scatter(t_test[0], statepl * np.exp(self.NN.u_ini[1]), color='k', alpha=0.5, s=50)


        ax_bottom.set_xlabel('Iteration')
        ax_bottom.set_ylabel('Total Loss')
        ax_bottom.set_yscale('log')
        ax_bottom.set_xlim(0, self.loss_list['iter'][-1])
        ax_bottom.set_ylim(1e-8, 1e2)
        ax_bottom.set_title("Total Loss")

        # **Initialize empty plots**
        theta_line, = axes[1].plot(t_test, np.empty_like(t_test), color='blue')
        v_line, = axes[0].plot(t_test, np.empty_like(t_test), color='red')
        loss_line, = ax_bottom.plot([], [], 'k-', label = "Total Loss")  # Initially empty
        ini_line, = ax_bottom.plot([], [], 'r-', label = "Initial Loss")  # Initially empty
        ode_line, = ax_bottom.plot([], [], 'b-', label = "ODE Loss")  # Initially empty

        # Add legend below the bottom plot with three columns
        ax_bottom.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=12, frameon=False)

        # **Preallocate total loss highlight point**
        loss_point = ax_bottom.scatter([], [], color='red', zorder=3)
        ini_point = ax_bottom.scatter([], [], color='red', zorder=3)
        ode_point = ax_bottom.scatter([], [], color='red', zorder=3)

        

        # **Create annotation - but we'll update it in each frame**
        loss_annotation = ax_bottom.annotate("", xy=(0, 0), xytext=(10, 10),
                                            textcoords="offset points",
                                            ha='center', fontsize=10,
                                            color='red', bbox=dict(boxstyle="round,pad=0.3",
                                                                fc="white", ec="red", alpha=0.8))
        
        ini_annotation = ax_bottom.annotate("", xy=(0, 0), xytext=(10, 10),
                                            textcoords="offset points",
                                            ha='center', fontsize=10,
                                            color='red', bbox=dict(boxstyle="round,pad=0.3",
                                                                fc="white", ec="red", alpha=0.8))
        
        ode_annotation = ax_bottom.annotate("", xy=(0, 0), xytext=(10, 10),
                                            textcoords="offset points",
                                            ha='center', fontsize=10,
                                            color='red', bbox=dict(boxstyle="round,pad=0.3",
                                                                fc="white", ec="red", alpha=0.8))

        # **Set fixed y-limits for all axes**
        axes[0].set_ylim(1e-10, 1e-7)
        axes[1].set_ylim(1e5, 5e7)

        def update(frame):
            iterations = self.loss_list["iter"][:frame+1]
            total_loss = self.loss_list["total"][:(frame+1)*iter_skip:iter_skip]
            ini_loss = self.loss_list["ini"][:(frame+1)*iter_skip:iter_skip]
            ode_loss = self.loss_list["ode"][:(frame+1)*iter_skip:iter_skip]
            last_x = iterations[-1]
            y_total, y_ini, y_ode= total_loss[-1], ini_loss[-1], ode_loss[-1]

            # **Use set_ydata() only**
            theta_line.set_ydata(statepl * np.exp(self.loss_list['q'][frame]))
            v_line.set_ydata( vpl * np.exp(self.loss_list['p'][frame]))

            # **Update loss plot efficiently**
            loss_line.set_data(iterations, total_loss)
            ini_line.set_data(iterations, ini_loss)
            ode_line.set_data(iterations, ode_loss)

            # **Update loss highlight point efficiently**
            loss_point.set_offsets(np.array([[last_x, y_total]]))
            ini_point.set_offsets(np.array([[last_x, y_ini]]))
            ode_point.set_offsets(np.array([[last_x, y_ode]]))

            # **Update total loss annotation**
            loss_annotation.xy = (last_x, y_total)
            loss_annotation.set_text(f'{y_total:.3e}')

            ini_annotation.xy = (last_x, y_ini)
            ini_annotation.set_text(f'{y_ini:.3e}')

            ode_annotation.xy = (last_x, y_ode)
            ode_annotation.set_text(f'{y_ode:.3e}')

            # Make sure all annotations are visible
            for ann in [loss_annotation]:
                if not hasattr(ann, '_visible'):
                    ann.set_visible(True)

            # Return all artists that need to be redrawn - added ICp_annotation and ICq_annotation
            return theta_line, v_line, loss_line, loss_point, loss_annotation, ini_annotation, ode_annotation

        # Critical changes:
        # 1. Added all annotations to the return tuple
        # 2. Set blit=False to ensure entire figure redraws
        ani = FuncAnimation(fig, update, frames=len(self.loss_list['q']), repeat=False, blit=False)

        # Ensure tight layout before saving
        plt.tight_layout()


        # Save animation
        writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("training_animation.mp4", writer=writer)

 
    def animate_layer_activations(loss_list, t_test, final_output=None, save_path="layer_activations.mp4"):
        plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\ffmpeg.exe'

        num_layers = len(loss_list['layer_activations'])
        ncols = 4
        nrows = int(np.ceil(num_layers / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
        axes = axes.flatten()

        def update(frame):
            for i, layer in enumerate(loss_list['layer_activations']):
                ax = axes[i]
                ax.clear()

                data = loss_list['layer_activations'][layer][frame].detach().numpy() 
                
                ax.plot(t_test, data)

                if final_output is not None:
                    ax.plot(t_test, final_output, 'k', linewidth=1)

                ax.set_title(f'Layer {layer}')

            # Clear any unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].clear()
                axes[j].axis('off')

        num_frames = len(loss_list['layer_activations'][list(loss_list['layer_activations'].keys())[0]])
        ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)
        
        writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)
        plt.close(fig)

    def plot_activations(self):

        PINN = self.NN

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns
        axes = axes.flatten()  # Flatten the 2D array to 1D for easier indexing
        final = PINN.forward(PINN.time).detach().numpy()

        for i in range(1, 9):
            lx = PINN.forward_by_layer(PINN.time, i).detach().numpy()
            axes[i - 1].plot(PINN.time, lx)
            axes[i - 1].set_title(f'Layer {i}')
            axes[i - 1].plot(PINN.time, final, 'k')

        plt.tight_layout()
        plt.show()