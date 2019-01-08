from tensorboardX import SummaryWriter
import time

log_dir = "tmp/tensorboardX_tester_" + str(time.time())
writer = SummaryWriter(log_dir=log_dir)

def get_str(dict):
    s = ""
    for key in dict:
        s += key + "=" + str(dict[key]) + "\n"
    return s

hyperparams = get_str({
    "learning_rate": 0.01,
    "freeze": True,
    "embedding_size": 300
})
print(hyperparams)
writer.add_text("hyperparams", hyperparams)
writer.add_text("str", "hi I'm here")



# writer.add_scalars("hyperparameters", {
#     "learning_rate": 0.001,
#     "freeze": True,
#     "embedding_size": 300
# })



writer.close()

print("TensorboardX output at:", log_dir)