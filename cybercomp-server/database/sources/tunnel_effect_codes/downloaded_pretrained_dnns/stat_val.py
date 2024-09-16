import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from cliffs_delta import cliffs_delta
from scipy import stats
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")


def get_mean_ci(group):
    mean_retain = group.mean()
    ci_low, ci_high = mean_confidence_interval(group)[1:]
    return mean_retain, ci_low, ci_high


def prepare_df():
    """
    Prepare dataframes for analysis
    """
    df = pd.read_excel("tunnel_effect.xlsx")
    Imagenet_df = df[df["ID Dataset"] == "Imagenet-100"]
    resolution_df = Imagenet_df[Imagenet_df["DNN Arch."].isin(["VGGm-17", "ViT-T+", "ResNet-18"])]

    # resolution
    resolution_32_df = resolution_df[resolution_df["Resolution"] == 32]
    resolution_64_df = resolution_df[resolution_df["Resolution"] == 64]
    resolution_128_df = resolution_df[resolution_df["Resolution"] == 128]
    resolution_224_df = resolution_df[resolution_df["Resolution"] == 224]

    # augmentation
    augmnetation_off_df = df[df["Augmentations"] == 0]
    augmentation_on_df = df[df["Augmentations"] == 2]

    # stem
    # Imagenet-100 & resolution 32 224
    # VGGm-11, VGGm-17, ViT-T, ViT-T+, ResNet-18, ResNet-34
    stem_df = df[(df["ID Dataset Size"] == 126689) & (df["Resolution"].isin([32, 224]))]
    stem_df = stem_df[
        stem_df["DNN Arch."].isin(["VGGm-11", "VGGm-17", "ViT-T", "ViT-T+", "ResNet-18", "ResNet-34"])
    ]
    group_3 = stem_df[stem_df["Stem"] == 3]
    group_7 = stem_df[stem_df["Stem"] == 7]
    group_8 = stem_df[stem_df["Stem"] == 8]

    # spatial reduction
    # Imagenet-100 & resolution 32
    # VGGm-11, VGGm-17 vs VGGmdag-11, VGGmdag-17
    sr_df = df[
        ((df["ID Dataset Size"] == 126689) | (df["ID Dataset"] == "cifar-100"))
        & (df["Resolution"].isin([32]))
    ]

    sr_df_1 = sr_df[sr_df["DNN Arch."].isin(["VGGmdag-11", "VGGmdag-17"])]
    sr_df_05 = sr_df[sr_df["DNN Arch."].isin(["VGGm-11", "VGGm-17"])]

    # depth
    # Imagenet-100 & resolution 32 224
    depth_df = df[(df["ID Dataset Size"] == 126689) & (df["Resolution"].isin([32, 224]))]
    depth_df = depth_df[
        depth_df["DNN Arch."].isin(["VGGm-11", "VGGm-17", "ResNet-18", "ResNet-34", "ViT-T", "ViT-T+"])
    ]
    df_11 = depth_df[depth_df["Depth"] == 11]
    df_12 = depth_df[depth_df["Depth"] == 12]
    df_17 = depth_df[depth_df["Depth"] == 17]

    df_18 = depth_df[depth_df["Depth"] == 18]
    df_18_ResNet = df_18[df_18["DNN Arch."] == "ResNet-18"]
    df_18_vit = df_18[df_18["DNN Arch."] == "ViT-T+"]
    df_34 = depth_df[depth_df["Depth"] == 34]

    # CNN vs Transformer
    # architecture_df = df[(df["ID Dataset Size"] == 126689) & (df["Resolution"].isin([32, 224]))]
    architecture_df = df[(df["ID Dataset Size"] == 126689) & (df["Resolution"].isin([32, 64, 128, 224]))]
    cnn_df = architecture_df[architecture_df["CNN vs ViT"] == "CNN"]
    cnn_vgg_df = cnn_df[cnn_df["DNN Arch."].isin(["VGGm-11", "VGGm-17"])]
    cnn_resnet_df = cnn_df[cnn_df["DNN Arch."].isin(["ResNet-18", "ResNet-34"])]

    transformer_df = architecture_df[architecture_df["CNN vs ViT"] == "Transformer"]

    # overparam level
    # Imagenet-100 & resolution 32 224

    overparam_df = df[(df["ID Dataset Size"] == 126689) & (df["Resolution"].isin([32, 224]))]
    overparam_df = overparam_df[
        overparam_df["DNN Arch."].isin(["VGGm-11", "VGGm-17", "ViT-T", "ViT-T+", "ResNet-18", "ResNet-34"])
    ]
    overp_44_df = overparam_df[overparam_df["OverParam. Level"] == 44.28166613]
    overp_66_df = overparam_df[overparam_df["OverParam. Level"] == 66.22516556]

    overp_74_df = overparam_df[overparam_df["OverParam. Level"] == 74.67104484]
    overp_88_df = overparam_df[overparam_df["OverParam. Level"] == 88.56333225]
    overp_158_df = overparam_df[overparam_df["OverParam. Level"] == 158.49837]
    overp_168_df = overparam_df[overparam_df["OverParam. Level"] == 168.3650514]

    # ID Dataset
    id_dataset_df = df[df["DNN Arch."].isin(["VGGm-11"])]
    id_dataset_df = id_dataset_df[id_dataset_df["Resolution"] == 32]
    id_dataset_df = id_dataset_df[id_dataset_df["OverParam. Level"].isin([189.2, 74.67104484])]
    # remove Imagenet experiments for varying class and sample size
    id_dataset_df = id_dataset_df[
        ~((id_dataset_df["ID Dataset"] == "Imagenet-100") & (id_dataset_df["OverParam. Level"] == 189.2))
    ]
    imagnet_df = id_dataset_df[id_dataset_df["ID Dataset"] == "Imagenet-100"]
    cifar_10_df = id_dataset_df[id_dataset_df["ID Dataset"] == "cifar-10"]
    cifar_100_df = id_dataset_df[id_dataset_df["ID Dataset"] == "cifar-100"]

    # dataset size & class
    sample_calss_df = df[(df["ID Dataset Size"] != 126689) & (df["ID Dataset"] == "Imagenet-100")]
    c10_s1000_df = sample_calss_df[sample_calss_df["ID Class Count"] == 10]
    c50_s200_df = sample_calss_df[sample_calss_df["ID Class Count"] == 50]
    c100_s100_df = sample_calss_df[
        (sample_calss_df["ID Class Count"] == 100) & (sample_calss_df["ID Dataset Size"] == 10000)
    ]
    c100_s200_df = sample_calss_df[
        (sample_calss_df["ID Class Count"] == 100) & (sample_calss_df["ID Dataset Size"] == 20000)
    ]
    c100_s500_df = sample_calss_df[
        (sample_calss_df["ID Class Count"] == 100) & (sample_calss_df["ID Dataset Size"] == 50000)
    ]
    c100_s700_df = sample_calss_df[
        (sample_calss_df["ID Class Count"] == 100) & (sample_calss_df["ID Dataset Size"] == 70000)
    ]

    dfs = {
        "Resolution": {
            "$32^2$": resolution_32_df,
            "$64^2$": resolution_64_df,
            "$128^2$": resolution_128_df,
            "$224^2$": resolution_224_df,
        },
        "Augmentation": {"off": augmnetation_off_df, "on": augmentation_on_df},
        "Stem": {"$3$": group_3, "$7$": group_7, "$8$": group_8},
        "Spatial Reduction": {"$1$": sr_df_1, "$0.5$": sr_df_05},
        "DNN Arch.": {
            # "CNN": cnn_df,
            "CNN(VGG)": cnn_vgg_df,
            "CNN(ResNet)": cnn_resnet_df,
            "Transformer": transformer_df,
        },
        "Depth": {
            "$11$": df_11,
            "$12$": df_12,
            "$17$": df_17,
            "$18(ResNet-18)$": df_18_ResNet,
            "$18(ViT-T+)$": df_18_vit,
            "$34$": df_34,
        },
        "Overparam Level": {
            "$44.28$": overp_44_df,
            "$66.23$": overp_66_df,
            "$74.67$": overp_74_df,
            "$88.56$": overp_88_df,
            "$158.5$": overp_158_df,
            "$168.37$": overp_168_df,
        },
        "ID Dataset": {"Imagenet-100": imagnet_df, "cifar-10": cifar_10_df, "cifar-100": cifar_100_df},
        "Sample Class": {
            "c10\_s1000": c10_s1000_df,
            "c50\_s200": c50_s200_df,
            "c100\_s100": c100_s100_df,
            "c100\_s200": c100_s200_df,
            "c100\_s500": c100_s500_df,
            "c100\_s700": c100_s700_df,
        },
    }
    return dfs


def get_wilcoxon_metrics(group1, group2):
    # compute wilcoxon signed-rank test for paired samples
    _, ood_retained_p_value = wilcoxon(group1["Percentage OOD retained"], group2["Percentage OOD retained"])
    _, pearson_p_value = wilcoxon(group1["Pearson Correlation"], group2["Pearson Correlation"])
    _, id_ood_mult_p_value = wilcoxon(group1["ID OOD Alignment"], group2["ID OOD Alignment"])
    return ood_retained_p_value, pearson_p_value, id_ood_mult_p_value


effect_size_map = {
    "negligible": "N",
    "small": "S",
    "medium": "M",
    "large": "L",
}


def compute_wilcoxon_and_mean_effect(dfs, table=True):
    resolutions = list(dfs.keys())
    results = {}

    for res1, res2 in combinations(resolutions, 2):
        df1 = dfs[res1]
        df2 = dfs[res2]

        ood_retained_p_value, pearson_p_value, id_ood_p_value = get_wilcoxon_metrics(df1, df2)

        ood_retained_delta, ood_retained_effect_size = cliffs_delta(
            df1["Percentage OOD retained"], df2["Percentage OOD retained"]
        )
        pearson_delta, pearson_effect_size = cliffs_delta(
            df1["Pearson Correlation"], df2["Pearson Correlation"]
        )
        id_ood_delta, id_ood_effect_size = cliffs_delta(df1["ID OOD Alignment"], df2["ID OOD Alignment"])

        ood_retained_effect_size = effect_size_map[ood_retained_effect_size]
        pearson_effect_size = effect_size_map[pearson_effect_size]
        id_ood_effect_size = effect_size_map[id_ood_effect_size]

        ood_retained_delta = abs(ood_retained_delta)
        pearson_delta = abs(pearson_delta)
        id_ood_delta = abs(id_ood_delta)

        if table:
            if ood_retained_p_value < 0.001:
                ood_retained_p_value = "p<0.001"
            else:
                ood_retained_p_value = f"{ood_retained_p_value:.3f}"
            if pearson_p_value < 0.001:
                pearson_p_value = "p<0.001"
            else:
                pearson_p_value = f"{pearson_p_value:.3f}"
            if id_ood_p_value < 0.001:
                id_ood_p_value = "p<0.001"
            else:
                id_ood_p_value = f"{id_ood_p_value:.3f}"
        else:
            ood_retained_p_value = f"{ood_retained_p_value:.3f}"
            pearson_p_value = f"{pearson_p_value:.3f}"
            id_ood_p_value = f"{id_ood_p_value:.3f}"

        results[(res1, res2)] = {
            "OOD Retained p-value": ood_retained_p_value,
            "Pearson p-value": pearson_p_value,
            "ID-OOD p-value": id_ood_p_value,
            "delta OOD Retained": ood_retained_delta,
            "delta Pearson": pearson_delta,
            "delta ID-OOD": id_ood_delta,
            "Effect Size OOD Retained": ood_retained_effect_size,
            "Effect Size Pearson": pearson_effect_size,
            "Effect Size ID-OOD": id_ood_effect_size,
        }

    return results


def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h1, h2 = stats.t.interval(confidence, n, loc=m, scale=se)
    m = abs(m)
    return m, h1, h2


def get_metrics(data):
    """
    Args:
        data: pandas dataframe

    compute mean and 95% CI for Percentage OOD retained, Pearson Correlation, ID OOD Alignment
    """
    mean_ood_retain, l_ood_retain, h_ood_retain = mean_confidence_interval(data["Percentage OOD retained"])

    mean_pearson, l_pearson, h_pearson = mean_confidence_interval(data["Pearson Correlation"])

    mean_id_ood_mult, l_id_ood_mult, h_id_ood_mult = mean_confidence_interval(data["ID OOD Alignment"])

    return (
        mean_ood_retain,
        l_ood_retain,
        h_ood_retain,
        mean_pearson,
        l_pearson,
        h_pearson,
        mean_id_ood_mult,
        l_id_ood_mult,
        h_id_ood_mult,
    )


def print_variables_summary():
    dfs = prepare_df()
    for key, value in dfs.items():
        print(f"\\textcolor{{orange}}{{\\emph{{\\textbf{{{key}}}}}}} & & & & & & \\\\")

        for k, v in value.items():
            (
                mean_ood_retained,
                l_ood_retained,
                h_ood_retained,
                mean_pearson,
                l_pearson,
                h_pearson,
                mean_id_ood_mult,
                l_id_ood_mult,
                h_id_ood_mult,
            ) = get_metrics(v)
            oneline = f"{k} & ${mean_ood_retained:.2f}$ & ${l_ood_retained:.2f}-{h_ood_retained:.2f}$ & ${mean_pearson:.2f}$ & ${l_pearson:.2f}-{h_pearson:.2f}$ & ${mean_id_ood_mult:.2f}$ & ${l_id_ood_mult:.2f}-{h_id_ood_mult:.2f}$ \\\\"
            print(oneline)
        print("\hline")
        print()


def print_comparison_summary():
    dfs = prepare_df()
    for key, value in dfs.items():
        print(f"\\textcolor{{orange}}{{\\emph{{\\textbf{{{key}}}}}}} & & & & & & \\\\")
        results = compute_wilcoxon_and_mean_effect(value)

        for k, v in results.items():
            single_group_size = len(value[k[0]])

            effect_size_ood_retained = v["Effect Size OOD Retained"]
            effect_size_pearson = v["Effect Size Pearson"]
            effect_size_id_ood = v["Effect Size ID-OOD"]

            delta_ood_retained = v["delta OOD Retained"]
            delta_pearson = v["delta Pearson"]
            delta_id_ood = v["delta ID-OOD"]

            ood_retained_p_value = v["OOD Retained p-value"]
            pearson_p_value = v["Pearson p-value"]
            id_ood_p_value = v["ID-OOD p-value"]

            oneline = f"{k[0]} vs {k[1]} & ${delta_ood_retained:.3f}({effect_size_ood_retained})$ & ${ood_retained_p_value}$ & ${delta_pearson:.3f}({effect_size_pearson})$ & ${pearson_p_value}$ & ${delta_id_ood:.3f}({effect_size_id_ood})$ & ${id_ood_p_value}$ & {single_group_size} \\\\"
            print(oneline)
        print("\hline")


if __name__ == "__main__":
    print_variables_summary()
    print_comparison_summary()
