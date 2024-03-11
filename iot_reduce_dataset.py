import os              # loading files
import numpy as np     # intermediary data handling
import pandas as pd    # data processing

DATASET_DIRECTORY = "./data/unb-cic-iot-dataset/wataiData/csv/CICIoT2023/"
csv_files = [k for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')]
csv_files.sort()

def sample_rows(df, percent_rows, is_validation):
    '''
    Samples rows sequentially based on the specified percentage, without overlap between samples.
    Note that class balance is preserved. 
    
    Parameters
    ----------------------
    df (type: pd.DataFrame)
    percent_rows (type: float, range: 0-1)
    is_validation (type: bool): Flag to indicate if the sampling is for a validation set.
    
    
    Returns
    ----------------------
    pd.DataFrame
    - Contains percent_rows of each class in input df
    '''
    
    # Define the order and size of each percentage sample
    percentages_ordered = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    labels = df['label'].unique()
    dfs_condensed = []
    
    # Adjust for validation by setting a distinct position for validation percentage if needed
    validation_index = None
    if is_validation and percent_rows == 0.01:
        validation_index = len(percentages_ordered)  # Validation percentage is treated as an extra, distinct case
    
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    
    # Select rows with chosen label
    for label in labels:
        mask = df_shuffled['label'] == label
        df_by_label = df_shuffled[mask]
        
        # Calculate cumulative start indices for each percentage
        cumulative_indices = [int(sum(percentages_ordered[:i]) * len(df_by_label)) for i in range(len(percentages_ordered)+1)]
        
        if validation_index is not None:
            # For validation, calculate start and end based on the entire dataset minus previous samples
            start_idx = cumulative_indices[-1]  # Start after the last main percentage
            end_idx = start_idx + int(0.01 * len(df_by_label))  # Add 1% of the label-specific rows for validation
        else:
            # Find the index for the requested percentage
            index = percentages_ordered.index(percent_rows)
            start_idx = int(sum(percentages_ordered[:index]) * len(df_by_label))
            end_idx = int(sum(percentages_ordered[:index+1]) * len(df_by_label))
        
        # Sample the DataFrame based on calculated indices
        sampled_df = df_by_label.iloc[start_idx:end_idx]
        dfs_condensed.append(sampled_df)
    
    # Shuffle all samples
    return pd.concat(dfs_condensed, ignore_index = True).sample(frac = 1)


# Map IANA Protocol numbers to strings (leave user option to 1-hot or numerically encode)
iana_map = { 
    "0": "HOPOPT", "1": "ICMP", "2": "IGMP", "3": "GGP", "4": "IPv4", "5": "ST", 
    "6": "TCP", "7": "CBT", "8": "EGP", "9": "IGP", "10": "BBN-RCC-MON", "11": "NVP-II", 
    "12": "PUP", "13": "ARGUS (deprecated)", "14": "EMCON", "15": "XNET", "16": "CHAOS", 
    "17": "UDP", "18": "MUX", "19": "DCN-MEAS", "20": "HMP", "21": "PRM", "22": "XNS-IDP", 
    "23": "TRUNK-1", "24": "TRUNK-2", "25": "LEAF-1", "26": "LEAF-2", "27": "RDP", 
    "28": "IRTP", "29": "ISO-TP4", "30": "NETBLT", "31": "MFE-NSP", "32": "MERIT-INP", 
    "33": "DCCP", "34": "3PC", "35": "IDPR", "36": "XTP", "37": "DDP", "38": "IDPR-CMTP", 
    "39": "TP++", "40": "IL", "41": "IPv6", "42": "SDRP", "43": "IPv6-Route", 
    "44": "IPv6-Frag", "45": "IDRP", "46": "RSVP", "47": "GRE", "48": "DSR", "49": "BNA", 
    "50": "ESP", "51": "AH", "52": "I-NLSP", "53": "SWIPE (deprecated)", "54": "NARP", 
    "55": "MOBILE", "56": "TLSP", "57": "SKIP", "58": "IPv6-ICMP", "59": "IPv6-NoNxt", 
    "60": "IPv6-Opts", "62": "CFTP", "64": "SAT-EXPAK", "65": "KRYPTOLAN", "66": "RVD", 
    "67": "IPPC", "69": "SAT-MON", "70": "VISA", "71": "IPCV", "72": "CPNX", "73": "CPHB", 
    "74": "WSN", "75": "PVP", "76": "BR-SAT-MON", "77": "SUN-ND", "78": "WB-MON", 
    "79": "WB-EXPAK", "80": "ISO-IP", "81": "VMTP", "82": "SECURE-VMTP", "83": "VINES", 
    "84": "IPTM", "85": "NSFNET-IGP", "86": "DGP", "87": "TCF", "88": "EIGRP", 
    "89": "OSPFIGP", "90": "Sprite-RPC", "91": "LARP", "92": "MTP", "93": "AX.25", 
    "94": "IPIP", "95": "MICP (deprecated)","96": "SCC-SP", "97": "ETHERIP", "98": "ENCAP", 
    "100": "GMTP", "101": "IFMP", "102": "PNNI", "103": "PIM", "104": "ARIS", "105": "SCPS", 
    "106": "QNX", "107": "A/N", "108": "IPComp", "109": "SNP", "110": "Compaq-Peer", 
    "111": "IPX-in-IP", "112": "VRRP", "113": "PGM", "114": "", "115": "L2TP", "116": "DDX",  
    "117": "IATP", "118": "STP", "119": "SRP", "120": "UTI", "121": "SMP", 
    "122": "SM (deprecated)", "123": "PTP","124": "ISIS over IPv4", "125": "FIRE", 
    "126": "CRTP", "127": "CRUDP", "128": "SSCOPMCE", "129": "IPLT", "130": "SPS", 
    "131": "PIPE", "132": "SCTP",  "133": "FC", "134": "RSVP-E2E-IGNORE", 
    "135": "Mobility Header", "136": "UDPLite", "137": "MPLS-in-IP", "138": "manet", 
    "139": "HIP", "140": "Shim6", "141": "WESP", "142": "ROHC", "143": "Ethernet", 
    "144": "AGGFRAG", "145": "NSH"
}

def iana_convert(df):
    df["Protocol Type"] = df["Protocol Type"].apply(lambda num : iana_map[ str(int(num)) ])
    return df

# Convert to reduced space dtypes to save data
dtypes = {
        'flow_duration': np.float32,
        'Header_Length': np.uint32,
        'Protocol Type': str,
        'Duration': np.float32,
        'Rate': np.uint32,
        'Srate': np.uint32,
        'Drate': np.float32,
        'fin_flag_number': np.bool_,
        'syn_flag_number': np.bool_,
        'rst_flag_number': np.bool_,
        'psh_flag_number': np.bool_,
        'ack_flag_number': np.bool_,
        'ece_flag_number': np.bool_,
        'cwr_flag_number': np.bool_,
        'ack_count': np.float16,
        'syn_count': np.float16,
        'fin_count': np.uint16,
        'urg_count': np.uint16, 
        'rst_count': np.uint16, 
        'HTTP': np.bool_, 
        'HTTPS': np.bool_, 
        'DNS': np.bool_, 
        'Telnet': np.bool_,
        'SMTP': np.bool_, 
        'SSH': np.bool_, 
        'IRC': np.bool_, 
        'TCP': np.bool_, 
        'UDP': np.bool_, 
        'DHCP': np.bool_, 
        'ARP': np.bool_, 
        'ICMP': np.bool_, 
        'IPv': np.bool_, 
        'LLC': np.bool_,
        'Tot sum': np.float32, 
        'Min': np.float32, 
        'Max': np.float32, 
        'AVG': np.float32, 
        'Std': np.float32, 
        'Tot size': np.float32, 
        'IAT': np.float32, 
        'Number': np.float32,
        'Magnitue': np.float32, 
        'Radius': np.float32, 
        'Covariance': np.float32, 
        'Variance': np.float32, 
        'Weight': np.float32, 
        'label': str
    }

def convert_dtype(df):
    # Adjust data type
    for col,typ in dtypes.items():
        df[col] = df[col].astype(typ)   
    
    # Format column names to lowercase snake
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Fix spelling error in original dataset
    df['magnitude'] = df['magnitue']
    return df.drop(['magnitue'], axis=1)

# Create raw label maps, simplified label maps, and cyberattack/no cyberattack label maps
reduced_labels = {
    'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS', 
    'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS', 
    'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS', 
    'DDoS-UDP_Fragmentation': 'DDoS', 'DDoS-ICMP_Fragmentation': 'DDoS', 
    'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS', 'DoS-UDP_Flood': 'DoS', 
    'DoS-SYN_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS', 
    'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai', 'Mirai-udpplain': 'Mirai', 
    'Recon-PingSweep': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon', 
    'VulnerabilityScan': 'Recon', 'Recon-HostDiscovery': 'Recon', 'DNS_Spoofing': 'Spoofing', 
    'MITM-ArpSpoofing': 'Spoofing', 'BenignTraffic': 'Benign', 'BrowserHijacking': 'Web', 
    'Backdoor_Malware': 'Web', 'XSS': 'Web', 'Uploading_Attack': 'Web', 'SqlInjection': 'Web', 
    'CommandInjection': 'Web', 'DictionaryBruteForce': 'BruteForce'
}


def label_map(df_34):
    df_8 = df_34.copy()
    df_2 = df_34.copy()
    
    # Adjust label classes
    df_8['label'] = df_8['label'].apply(lambda attack_name : reduced_labels[attack_name])
    df_2['benign'] = df_2['label'] == 'BenignTraffic'
    df_2 = df_2.drop(['label'], axis=1)
    
    return df_34,df_8,df_2

def write_helper(dfs, filename, append=True):
    """ Creates or appends """
    
    df_34,df_8,df_2 = dfs
    
    if append:
        df_34.to_csv(filename+'_34classes.csv', mode='a', index=False, header=False)
        df_8.to_csv(filename+'_8classes.csv', mode='a', index=False, header=False)
        df_2.to_csv(filename+'_2classes.csv', mode='a', index=False, header=False)
    else:
        df_34.to_csv(filename+'_34classes.csv', index=False)
        df_8.to_csv(filename+'_8classes.csv', index=False)
        df_2.to_csv(filename+'_2classes.csv', index=False)

def combine_csv(csv_files, percent, train_test):
    '''
    For memory-efficiency, extracts rows from one original CSV at a time. 
    Then, combines the rows into a larger CSV.
    
    Parameters
    ---------------------
    csv_files (type: list): Filepaths to the raw CSV files
    percent (type: float): The percentage of rows to sample from each CSV.
    train_test (type: bool): Flag indicating whether the operation is for training or testing (validation).
    
    
    Returns
    ---------------------
    None
    - Outputs to CSV file instead
    '''
    num = 30
    
    # File suffix based on train_test flag
    suffix = "" if train_test else "_validation"
    output_path = f'./kaggle/working/{percent}percent{suffix}'
    
    # Initialize output CSV with the first batch
    first_batch_files = csv_files[:num]
    first_batch_dfs = [pd.read_csv(DATASET_DIRECTORY + csv_file) for csv_file in first_batch_files]
    first_concat_df = pd.concat(first_batch_dfs, ignore_index=True)
    
    # Preprocessing and sampling for the first batch
    first_dfs = label_map(convert_dtype(iana_convert(sample_rows( 
        first_concat_df, percent_rows=percent, is_validation=not train_test
    ))))
    write_helper(first_dfs, output_path, append=False)  # Initialize the CSV file with the first batch
    del first_dfs, first_batch_dfs, first_concat_df

    print(f"Initialized {output_path} with the first batch.")
    
    # Process the remaining files in batches of num
    for i in range(num, len(csv_files), num):
        batch_files = csv_files[i:i+num]
        batch_dfs = [pd.read_csv(DATASET_DIRECTORY + csv_file) for csv_file in batch_files]
        concatenated_df = pd.concat(batch_dfs, ignore_index=True)
        
        # Preprocessing and sampling for the current batch
        dfs = label_map(convert_dtype(iana_convert(sample_rows( 
            concatenated_df, percent_rows=percent, is_validation=not train_test
        ))))
        
        # Append processed data to CSV
        write_helper(dfs, output_path)
        del dfs, batch_dfs, concatenated_df
        
        print(f"Appended batch {i // num} to {output_path}.")

    print("Finished processing all files.")

# sampling for training and testing datasets
for percent in [0.001, 0.005, 0.01, 0.05, 0.1]:
    combine_csv(csv_files, percent=percent,train_test=True)

# sampling for validation dataset
combine_csv(csv_files, percent=0.01, train_test=False)