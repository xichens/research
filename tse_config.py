feature = {
'list': ['tenure', 'has_wireless', 'mobile_subscribers', 'has_internet', 'wls_billing_90day', 'wln_billing_90day', \
        'internet_download_90day', 'internet_upload_90day', 'svod_watch_time_90day', 'home_phone_usage_90day', \
        'inbound_call_count_90day', 'wls_download_90day', 'wls_upload_90day'],
'model1': ['tenure', 'has_phone', 'MultipleLines', 'has_internet', 'usage_1', 'usage_2', 'usage_3', 'service_call', 'revenue'],
'model2': ['tenure', 'revenue', 'usage_3', 'usage_2', 'usage_1', 'service_call'],
'model3': ['tenure', 'usage_1', 'usage_2', 'usage_3', 'usage_4', 'service_call'],
}

feature_process = {
'fix': {'tenure': 'tenure',
        'tv_usage': 'svod_watch_time_90day',
        'phone_usage': 'home_phone_usage_90day',
        'service_call': 'inbound_call_count_90day',
        'has_phone': 'has_wireless',
        'has_internet': 'has_internet',
       },

'avg': {'revenue': ['wls_billing_90day', 'wln_billing_90day'],
        'internet_usage': ['internet_download_90day', 'internet_upload_90day'],
        'data_usage': ['wls_download_90day', 'wls_upload_90day'],
       },

'flag': {'MultipleLines': 'mobile_subscribers',},
'norm': ['tenure', 'revenue', 'tv_usage', 'phone_usage', 'internet_usage', 'data_usage', 'service_call'],
}

prod_feature = {
'tv': {'tenure': 'tenure',
       'has_phone': 'has_phone',
       'MultipleLines': 'MultipleLines',
       'has_internet': 'has_internet',
       'revenue': 'revenue',
       'usage_1': 'tv_usage',
       'usage_2': 'internet_usage',
       'usage_3': 'phone_usage',
       'usage_4': 'data_usage',
       'service_call': 'service_call',
      },

'internet': {'tenure': 'tenure',
             'has_phone': 'has_phone',
             'MultipleLines': 'MultipleLines',
             'has_internet': 'has_internet',
             'revenue': 'revenue',
             'usage_1': 'internet_usage',
             'usage_2': 'data_usage',
             'usage_3': 'phone_usage',
             'usage_4': 'tv_usage',
             'service_call': 'service_call',
            },

'home_phone': {'tenure': 'tenure',
               'has_phone': 'has_phone',
               'MultipleLines': 'MultipleLines',
               'has_internet': 'has_internet',
               'revenue': 'revenue',
               'usage_1': 'phone_usage',
               'usage_2': 'tv_usage',
               'usage_3': 'internet_usage',
               'usage_4': 'data_usage',
               'service_call': 'service_call',
              },

'wireless': {'tenure': 'tenure',
             'has_phone': 'has_phone',
             'MultipleLines': 'MultipleLines',
             'has_internet': 'has_internet',
             'revenue': 'revenue',
             'usage_1': 'phone_usage',
             'usage_2': 'data_usage',
             'usage_3': 'internet_usage',
             'usage_4': 'tv_usage',
             'service_call': 'service_call',
            },
}
