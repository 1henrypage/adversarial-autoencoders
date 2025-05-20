from matplotlib.ticker import MaxNLocator
from IPython import display
import matplotlib.pyplot as plt
from IPython.display import display as ipy_display
from tqdm import tqdm
class AAEExperimentBase:
    """
    Abstract experiment class for AAE: provides logging and flexible plotting.
    Set `plot_separately=True` to get one plot per loss.
    """
    def __init__(self, plot_separately=False):
        self.plot_separately = plot_separately
        self.plot_handle = None       # ← NEW
        self.acc_handle = None        # ← NEW (for accuracy if needed separately)

        self.recon_losses = []
        self.disc_cat_losses = []
        self.gen_cat_losses = []
        self.disc_style_losses = []
        self.gen_style_losses = []
        self.gen_cat_losses = []
        self.semi_supervised_losses = []
        self.val_accuracies = []

    def log_epoch(self, recon, disc_cat, gen_cat, disc_style, gen_style, semi_sup, val_acc):
        self.recon_losses.append(recon)
        self.disc_cat_losses.append(disc_cat)
        self.gen_cat_losses.append(gen_cat)
        self.disc_style_losses.append(disc_style)
        self.gen_style_losses.append(gen_style)
        self.semi_supervised_losses.append(semi_sup)
        self.val_accuracies.append(val_acc)

    def init_plot_handles(self):
        """
        Manually initializes plot handles after tqdm shows up.
        Call this once at the start of training (after tqdm prints).
        """
        if self.plot_separately:
            # Dummy first plot
            fig, _ = plt.subplots()
            self.plot_handle = ipy_display(fig, display_id=True)
            plt.close(fig)

            fig_acc, _ = plt.subplots()
            self.acc_handle = ipy_display(fig_acc, display_id=True)
            plt.close(fig_acc)
        else:
            fig, _ = plt.subplots()
            self.plot_handle = ipy_display(fig, display_id=True)
            plt.close(fig)

    def dummy_loop(self, epoch):
        if epoch != 0: return
        dummy_loop = tqdm(total=1, desc="Initializing plot area", leave=False)
        dummy_loop.update(1)
        dummy_loop.close()
        self.init_plot_handles()

    def plot_training(self, update=True):
        if len(self.recon_losses) < 2:
            return

        if self.plot_separately:
            fig, axs = plt.subplots(2, 3, figsize=(18, 8))
            axs = axs.flatten()

            losses = [
                (self.recon_losses, 'Recon Loss'),
                (self.disc_cat_losses, 'Disc Cat Loss'),
                (self.gen_cat_losses, 'Gen Cat Loss'),
                (self.disc_style_losses, 'Disc Style Loss'),
                (self.gen_style_losses, 'Gen Style Loss'),
                (self.semi_supervised_losses, 'Semi-Sup Loss')
            ]

            for ax, (loss, title) in zip(axs, losses):
                ax.plot(loss)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            fig.suptitle("Training Losses", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            fig_acc, ax_acc = plt.subplots(figsize=(7, 5))
            ax_acc.plot(self.val_accuracies, label='Validation Accuracy')
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy (%)")
            ax_acc.set_title("Validation Accuracy")
            ax_acc.legend()
            ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))

            if update:
                if self.plot_handle is None:
                    self.plot_handle = display.display(fig, display_id=True)
                else:
                    self.plot_handle.update(fig)

                if self.acc_handle is None:
                    self.acc_handle = display.display(fig_acc, display_id=True)
                else:
                    self.acc_handle.update(fig_acc)

            plt.close(fig)
            plt.close(fig_acc)

        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            axs[0].plot(self.recon_losses, label='Recon Loss')
            axs[0].plot(self.disc_cat_losses, label='Disc Cat Loss')
            axs[0].plot(self.gen_cat_losses, label='Gen Cat Loss')
            axs[0].plot(self.disc_style_losses, label='Disc Style Loss')
            axs[0].plot(self.gen_style_losses, label='Gen Style Loss')
            axs[0].plot(self.semi_supervised_losses, label='Semi-Sup Loss')
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].set_title("Training Losses")
            axs[0].legend()
            axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

            axs[1].plot(self.val_accuracies, label='Validation Accuracy')
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Accuracy (%)")
            axs[1].set_title("Validation Accuracy")
            axs[1].legend()
            axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

            plt.tight_layout()

            if update:
                if self.plot_handle is None:
                    self.plot_handle = display.display(fig, display_id=True)
                else:
                    self.plot_handle.update(fig)

            plt.close(fig)
