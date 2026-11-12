import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("data/processed/wisdm.npz")
X, y, sid = d["X"], d["y"], d["subject_id"]
names = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing"]

# pick one user who has all 5 activities
user = next(u for u in np.unique(sid) if len(np.unique(y[sid == u])) == 5)
t = np.arange(50) / 20                      # 50 samples at 20 Hz = 2.5 seconds

fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
for c, ax in enumerate(axes):
    w = X[(sid == user) & (y == c)][0]      # first window of this activity
    for k, axis in enumerate("xyz"):
        ax.plot(t, w[:, k], label=axis)
    ax.set_ylabel(names[c] + "\n(g)")
    ax.legend(loc="upper right", ncol=3)
axes[-1].set_xlabel("time (s)")
fig.suptitle(f"WISDM: one 2.5 s window per activity (user {user})")
plt.tight_layout()
plt.savefig("wisdm_windows.png", dpi=150)
print("saved wisdm_windows.png, user", user)