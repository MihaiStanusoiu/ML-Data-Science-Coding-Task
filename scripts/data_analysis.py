import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_statistics(data: pd.DataFrame) -> pd.DataFrame:
    return data.describe()

def analyze_pv_history(dataset: pd.DataFrame):
    # group by day: split data into days by timestamp, only from 08:00 to 16:00
    # dataset.pv_history = dataset.pv_history[dataset.pv_history['time'].dt.hour.between(8, 16)]


    # dataset.pv_history["month"] = dataset.pv_history["time"].dt.month
    pv_history_stats = compute_statistics(dataset['pv_generation'])
    plot_timestamped_data(data=dataset, title="PV Generation Data Distribution by Hour", x_key="hour",
                          y_key="pv_generation", stats=pv_history_stats)
    # plot_timestamped_data(data=dataset.pv_history, title="PV Generation Data Distribution by Month", x_key="month",
    #                       y_key="pv_generation", stats=pv_history_stats)

def analyze_weather_measurements(dataset: pd.DataFrame):
    stats = compute_statistics(dataset)

    plot_timestamped_data(data=dataset, title="Global Horizontal Irradiation Measurement Data Distribution", x_key="hour", y_key="cglo", stats=stats)
    plot_timestamped_data(data=dataset, title="Wind Speed Measurement Data Distribution", x_key="hour", y_key="ff", stats=stats)
    plot_timestamped_data(data=dataset, title="Air Temperature Measurement Data Distribution", x_key="hour", y_key="tl", stats=stats)


def plot_timestamped_data(data: pd.DataFrame, stats: pd.DataFrame, title: str, x_key: str, y_key: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x_key, y=y_key)

    # # Overlay mean + quantiles
    # plt.axvline(stats["mean"], color="red", linestyle="-", label=f"mean: {stats['mean']:.1f}")
    # plt.axvline(stats["50%"], color="green", linestyle="--", label=f"median: {stats['50%']:.1f}")
    # plt.axvline(stats["25%"], color="orange", linestyle="--", label=f"25%: {stats['25%']:.1f}")
    # plt.axvline(stats["75%"], color="orange", linestyle="--", label=f"75%: {stats['75%']:.1f}")

    plt.title(title)
    plt.legend()
    plt.show()

    # plt.hist(data, bins=nr_bins)
    # plt.title(title)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.show()

def plot_train_val_loss(train_losses: list, val_losses: list) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss over Epochs')
    plt.legend()
    plt.show()

def plot_predicted_actual(predictions: list, actual: list) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predicted', color='blue')
    plt.plot(actual, label='Actual', color='orange')
    plt.xlabel('Time')
    plt.ylabel('PV Generation')
    plt.title('Predicted vs Actual PV Generation')
    plt.legend()
    plt.show()