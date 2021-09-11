# Replicating Results for "Deep multitask learning of gene risk for comorbid neurodevelopmental disorders"

The results of DeepND for autism spectrum disorder (ASD) and intellectual disability (ID) are presented in ["Deep multitask learning of gene risk for comorbid neurodevelopmental disorders"](https://www.biorxiv.org/content/10.1101/2020.06.13.150201v3). To replicate the results, you /*must/* have a GPU Server access with at least 70GB memory. Then, a step by step tutorial is as follows:

1. Complete the initialization step as described in the [Installation Guide](https://github.com/ciceklab/DeepND/blob/master/README.md#installation-guide).
2. Open the *config* file and modify the following parameters.
   ```
   network=brainspan_all
   ```
   
3. Finally, run the following terminal command:
   ```
   python main.py
   ```
4. For more advanced setups that may require GPU parallelism, please refer to the descriptions in the *config* file.
5. To use generated results in your own work, please cite to the original manuscript.
```
@article{beyreli2021deep,
  title={Deep multitask learning of gene risk for comorbid neurodevelopmental disorders},
  author={Beyreli, Ilayda and Karakahya, Oguzhan and Cicek, A Ercument},
  journal={bioRxiv},
  pages={2020--06},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
