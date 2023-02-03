import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_hex


class NetflixAnalyzer:
    def __init__(self, data_folder, export_folder):
        self.path_to_data = self.search_for_data_file(data_folder)
        self.export_folder = export_folder
        self.data = self.get_data()

        # Color palettes
        self.color_palette_user = None
        self.color_palette_months = None
        self.color_palette_years = None
        self.color_palette_weekdays = None
        self.color_palette_time = None

    def search_for_data_file(self, data_folder):
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file == "ViewingActivity.csv":
                    return os.path.join(root, file)
        raise FileNotFoundError(
            f"ViewingActivity.csv not found in the directory tree starting from {data_folder}"
        )

    def get_data(self):
        return pd.read_csv(self.path_to_data)

    def create_export_folder(self):
        os.makedirs(self.export_folder, exist_ok=True)

    def get_interpolated_hex_codes(
        self, k, color1=[219, 0, 0, 255], color2=[0, 0, 0, 255]
    ):
        # Defaults to Netflix red and black
        colors = [color1, color2]

        # Interpolate between Netflix red and black in k steps
        cm = LinearSegmentedColormap.from_list("", np.array(colors) / 255, 256)
        hex_colors = [to_hex(cm(v)) for v in np.linspace(0, 1, k)]
        return hex_colors

    def get_color_palette(
        self, k, color1=[219, 0, 0, 255], color2=[0, 0, 0, 255], color3=None
    ):
        # Get hex colors (defaults to Netflix red and black)
        if color3 is None:
            # Interpolate between color1 and color2
            hex_colors = self.get_interpolated_hex_codes(
                k, color1=color1, color2=color2
            )
        else:
            # Interpolate between color1 and color3 and color2
            hex_colors1 = self.get_interpolated_hex_codes(
                k // 2, color1=color1, color2=color3
            )

            hex_colors2 = self.get_interpolated_hex_codes(
                k // 2, color1=color3, color2=color2
            )
            hex_colors = hex_colors1 + hex_colors2

        # Create palette
        color_palette = sns.color_palette(hex_colors)
        return color_palette

    def setup_color_palettes(self, n_years=None, n_time=None):
        # Years colors if provided
        if n_years is not None:
            self.color_palette_years = self.get_color_palette(n_years)

        # Time colors if provided
        if n_time is not None:
            self.color_palette_time = self.get_color_palette(
                n_time,
                color1=[0, 0, 0, 255],
                color2=[255, 255, 255, 255],
                color3=[219, 0, 0, 255],
            )

        # User colors
        n_user = len(self.data["Profile Name"].unique())
        self.color_palette_user = self.get_color_palette(n_user)

        # Months colors
        self.color_palette_months = self.get_color_palette(12)

        # Weekday colors
        self.color_palette_weekdays = self.get_color_palette(7)

    def process_data(self):
        # Clean data
        self.data = self.data.loc[self.data["Supplemental Video Type"].isnull()]

        # Cast data types
        self.data["Start Time"] = pd.to_datetime(self.data["Start Time"])
        self.data["Date"] = pd.to_datetime(self.data["Start Time"].dt.date)
        self.data["Duration"] = pd.to_timedelta(self.data["Duration"])

        # Get different features
        self.data["Duration in s"] = self.data.loc[:, "Duration"].astype(
            "timedelta64[s]"
        )
        self.data["Duration in m"] = self.data.loc[:, "Duration in s"] / 60
        self.data["Duration in h"] = self.data.loc[:, "Duration in m"] / 60

        self.data["Month"] = self.data["Start Time"].dt.month
        self.data["Year"] = self.data["Start Time"].dt.year
        self.data["Week"] = self.data["Start Time"].dt.isocalendar().week
        self.data["Weekday"] = self.data["Start Time"].dt.weekday
        self.data["Hour"] = self.data["Start Time"].dt.hour

    def create_all_days_dataframe(self):
        # Dataframe with watched hours for every single day since the first day - zero when no netflix is used
        df_date = self.data.groupby(["Profile Name", "Date"], as_index=False).sum(
            numeric_only=True
        )
        df_date_all_list = []

        # Do this for every user
        for name in self.data["Profile Name"].unique():
            df_name = df_date.query("`Profile Name`==@name")
            first_date = self.data.query("`Profile Name`==@name")["Date"].min().date()
            last_date = self.data.query("`Profile Name`==@name")["Date"].max().date()

            df_date_all = pd.DataFrame(
                pd.date_range(start=first_date, end=last_date), columns=["Date"]
            )
            df_date_all["Profile Name"] = name
            df_date_all["Duration in h"] = 0  # initialise every day with zeros hours

            # Insert actual watched hours when netflix is used
            for date in df_name.Date.unique():
                df_date_all.loc[
                    df_date_all.Date.isin([date]), "Duration in h"
                ] = df_name.query("Date==@date")["Duration in h"].item()

            df_date_all_list.append(df_date_all)

        # Combine all user dataframes
        df_date_all = pd.concat(df_date_all_list, ignore_index=True)
        df_date_all["Weekday"] = df_date_all["Date"].dt.weekday
        return df_date_all

    def plot_total_watched_content(self):
        # Dropping duplicates caused by pressing pause and such
        data_tmp = self.data.drop_duplicates(
            subset=["Title", "Profile Name", "Month", "Year", "Week", "Hour"]
        ).copy()
        g = sns.displot(
            data=data_tmp,
            x="Profile Name",
            hue="Profile Name",
            palette=self.color_palette_user,
            alpha=1,
        )
        g.set_xticklabels(rotation=30)
        g.fig.suptitle("Total watched movies/shows")
        plt.savefig(
            os.path.join(self.export_folder, "total_watched_content.png"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_total_watched_hours(self):
        g = sns.catplot(
            data=self.data,
            x="Profile Name",
            y="Duration in h",
            estimator=sum,
            errorbar=None,
            order=self.data.groupby(["Profile Name"])
            .sum(numeric_only=True)
            .sort_values("Duration in h", ascending=False)
            .index,
            palette=self.color_palette_user,
            saturation=1,
            kind="bar",
        )
        g.set_xticklabels(rotation=30)
        g.set_ylabels("Watch time in h")
        g.fig.suptitle("Total watched hours")
        plt.savefig(
            os.path.join(self.export_folder, "total_watched_hours.png"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_watched_hours_over_years(self):
        # Plot hours over the year for each user
        for name in self.data["Profile Name"].unique():
            df_name = self.data.query("`Profile Name`==@name")
            n_years = len(df_name.Year.unique())
            self.setup_color_palettes(n_years)
            plt.figure()
            sns.catplot(
                data=df_name,
                x="Year",
                y="Duration in h",
                estimator=sum,
                errorbar=None,
                kind="bar",
                palette=self.color_palette_years,
                saturation=1,
            )
            plt.ylabel("Watch time in h")
            plt.title(f"Watched hours over years for {name}")
            plt.savefig(
                os.path.join(
                    self.export_folder, f"watched_hours_over_years_{name}.png"
                ),
                bbox_inches="tight",
            )
            plt.close()

    def plot_mean_watched_hours_per_year(self):
        df_year = self.data.groupby(["Profile Name", "Year"], as_index=False).sum(
            numeric_only=True
        )
        g = sns.catplot(
            data=df_year,
            x="Profile Name",
            y="Duration in h",
            kind="bar",
            palette=self.color_palette_user,
            saturation=1,
            order=df_year.groupby(["Profile Name"])
            .mean()
            .sort_values("Duration in h", ascending=False)
            .index,
        )
        g.tick_params("x", rotation=30)
        plt.title("Mean watched hours per year")
        g.set_ylabels("Watch time in h")
        plt.savefig(
            os.path.join(self.export_folder, "mean_watched_hours_per_year.png"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_watched_hours_over_years_and_months(self):
        # Plot watched hours over years and months for each user
        for name in self.data["Profile Name"].unique():
            df_name = self.data.query("`Profile Name`==@name")
            fig, ax = plt.subplots(figsize=(20, 7))
            sns.barplot(
                data=df_name,
                x="Year",
                y="Duration in h",
                estimator=sum,
                errorbar=None,
                hue="Month",
                palette=self.color_palette_months,
                saturation=1,
                ax=ax,
            )
            ax.legend(title="Month", bbox_to_anchor=(1, 1), loc="upper left")
            ax.set(xlabel="Year", ylabel="Watch time in h")
            ax.set_title(f"Watched hours over years and months for {name}")
            plt.savefig(
                os.path.join(
                    self.export_folder,
                    f"watched_hours_over_years_and_months_{name}.png",
                ),
                bbox_inches="tight",
            )

            plt.close()

    def plot_watched_hours_over_weekdays(self, only_when_netflix_used=False):
        # Plot hours over weekdays for each user
        df_all_days = self.create_all_days_dataframe()
        # only for weekdays when netflix was actually used
        if only_when_netflix_used:
            df_all_days = df_all_days.query("`Duration in h`!=0")
            plot_name = "netflix days"
        else:
            plot_name = "all days"

        for name in self.data["Profile Name"].unique():
            df_name = df_all_days.query("`Profile Name`==@name")
            g = sns.catplot(
                data=df_name,
                x="Weekday",
                y="Duration in h",
                kind="bar",
                col="Profile Name",
                palette=self.color_palette_weekdays,
                saturation=1,
            )
            g.tick_params("x", rotation=30)
            g.set_xticklabels(
                [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
            )
            plt.subplots_adjust(top=0.85)
            plt.ylabel("Watch time in h")
            plt.title(f"Watched hours over weekdays for {name} on {plot_name}")
            plt.savefig(
                os.path.join(
                    self.export_folder,
                    f"watched_hours_over_weekdays_{name}_{plot_name}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

    def plot_time_over_weekdays_yearly_viewing_activity(self):
        weekdays = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        hours = np.arange(1, 25)
        for name in self.data["Profile Name"].unique():
            df_name = self.data.query("`Profile Name`==@name")
            # Count activities per weekday, hour and year
            df_name = df_name.groupby(
                ["Year", "Weekday", "Hour"], as_index=False
            ).count()
            # Average counted activites over years per weekday and hour
            df_name = df_name.groupby(["Weekday", "Hour"], as_index=False).mean()

            df_weekday_time = df_name.pivot_table(
                columns="Weekday", index="Hour", values="Start Time"
            )
            df_weekday_time = df_weekday_time.fillna(0)

            # Any missing hours?
            missing_hours = [h for h in range(24) if h not in df_weekday_time.index]
            if missing_hours:  # Is missing some hours
                missing_data = pd.DataFrame(
                    np.zeros((len(missing_hours), 7)), index=missing_hours
                )
                df_weekday_time = pd.concat(
                    [df_weekday_time, missing_data]
                ).sort_index()

            n_time = df_weekday_time.max().max()
            df_weekday_time = df_weekday_time.div(n_time)
            self.setup_color_palettes(n_time=100)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax = sns.heatmap(
                df_weekday_time,
                linewidth=0.5,
                cbar=True,
                cmap=self.color_palette_time,
                xticklabels=weekdays,
                yticklabels=hours,
            )
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
            ax.tick_params(axis="both", which="both", length=0)
            ax.xaxis.tick_top()
            ax.set_title(
                f"Normalized yearly average viewing activity over weekdays and time of day for {name}"
            )
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["Lowest", "Highest"])
            cbar.set_label("Activity")
            plt.savefig(
                os.path.join(
                    self.export_folder,
                    f"yearly_averaged_viewing_activity_over_weekdays_and_time_{name}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

    def analyze(self):
        self.create_export_folder()
        self.process_data()
        self.setup_color_palettes()

        self.plot_total_watched_content()
        self.plot_total_watched_hours()
        self.plot_mean_watched_hours_per_year()
        self.plot_watched_hours_over_years()
        self.plot_watched_hours_over_years_and_months()
        self.plot_watched_hours_over_weekdays(only_when_netflix_used=False)
        self.plot_watched_hours_over_weekdays(only_when_netflix_used=True)
        self.plot_time_over_weekdays_yearly_viewing_activity()


def main():
    data_folder = "data"
    export_folder = "results"
    na = NetflixAnalyzer(data_folder=data_folder, export_folder=export_folder)
    na.analyze()


if __name__ == "__main__":
    main()
