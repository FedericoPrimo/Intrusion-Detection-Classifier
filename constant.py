## Defining columns ##
CATEGORICAL_COLS = [
  'protocol_type', 'service', 'flag'
]
BOOLEAN_COLS = [
  'land', 'logged_in', 'lroot_shell',
  'lsu_attempted', 'is_host_login', 'is_guest_login'
]
CONTINUOUS_COLS = [
  'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment',
  'urgent', 'hot', 'num_failed_logins', 'lnum_compromised',
  'lnum_root', 'lnum_file_creations', 'lnum_shells',
  'lnum_access_files', 'lnum_outbound_cmds', 'count',
  'srv_count', 'dst_host_count', 'dst_host_srv_count'
]
PERCENTILES_COLS = [
  'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
  'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
  'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
  'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
  'dst_host_serror_rate', 'dst_host_srv_serror_rate',
  'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Columns to apply log transformation
TRANSFORM_TO_LOG = [
  'src_bytes', 'dst_bytes', 'count', 'srv_count'
]