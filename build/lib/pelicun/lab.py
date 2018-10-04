import json

def read_SimCenter_EDP_input_OLD(input_path, verbose=False):
    Demand_to_Acronym = dict([
        ('__type__', 'demand to acronym'),
        ('PFA', ['max_abs_acceleration', ]),
        ('PID', ['max_drift', ]),
        ('RD', ['residual_disp', ]),
    ])

    # initialize the dictionary of EDP data
    data = {}

    with open(input_path, 'r') as f:
        jd = json.load(f)['EngineeringDemandParameters']

    events = []
    for i, event_data in enumerate(jd):
        # to make sure every IM level has a unique and informative ID
        events.append('{}_{}'.format(i + 1, event_data['name']))

    for i, edp in enumerate(jd[0]['responses']):
        kind = _classify(Demand_to_Acronym, edp['type'])
        if kind not in data.keys():
            data.update({kind: []})

        scale_factor = 1.0  # we assume that EDPs are provided in standard units

        raw_data = dict([
            (event,
             np.array(jd[e]['responses'][i]['scalar_data'],
                      dtype=np.float64) * scale_factor
             )
            for e, event in enumerate(events)
        ])

        data[kind].append(dict(
            cline=int(edp['cline']),
            floor=int(edp['floor1' if kind == 'PID' else 'floor']),
            raw_data=raw_data,
            floor2=int(edp['floor2']) if kind == 'PID' else None,
        ))

    # print the parsed data to the screen if requested
    if verbose:
        for kind, EDP_list in data.items():
            print(kind)
            for EDP_attributes in EDP_list:
                for attribute, value in EDP_attributes.items():
                    if attribute is not 'raw_data':
                        print('\t', attribute, value)
                    else:
                        print('\t', attribute)
                        for event, edp_data in value.items():
                            print('\t\t', event,
                                  '| {} samples: '.format(len(edp_data)),
                                  '[{:.4f}, ..., {:.4f}]'.format(min(edp_data),
                                                                 max(edp_data)))
                print()
            print('-' * 75)

    return data

def read_SimCenter_DL_input_OLD(input_path, verbose=False):
    Occupancy_to_P58 = dict([
        ('__type__', 'occupancy to P58'),
        ('Commercial Office', ['office', ]),
        ('Elementary School', ['education',
                               'school']),
        ('Middle School', []),
        ('High School', []),
        ('Healthcare', ['healthcare', ]),
        ('Hospitality', ['hospitality',
                         'hotel']),
        ('Multi-Unit Residential', ['residence',
                                    'Residential']),
        ('Retail', ['retail', ]),
        ('Warehouse', ['warehouse',
                       'industrial']),
        ('Research Laboratories', ['research', ]),
    ])

    with open(input_path, 'r') as f:
        jd = json.load(f)

    # replace random values with their mean
    # we will load actual realizations of those values in a different method
    randoms = dict((rv['name'], rv['mean']) for rv in jd['RandomVariables'])
    jd = jd['GI']
    for attrib in jd.keys():
        if attrib in randoms:
            jd[attrib] = randoms[attrib]

    # load the other attributes
    # note that we assume that everything is provided in standard units
    data = dict(
        name=jd['name'],
        area=float(jd['area']) * m2,
        stories=int(jd['numStory']),
        year_built=int(jd['yearBuilt']),
        str_type=jd['structType'],
        occupancy=_classify(Occupancy_to_P58, jd['occupancy']),
        height=float(jd['height']) * m,
        replacement_cost=float(jd['replacementCost']),
        replacement_time=float(jd['replacementTime']),
    )
    if jd['population'] == 'auto':
        data.update(dict(population='auto'))
    else:
        data.update(dict(population=np.asarray(jd['population'])))

    # print the parsed data to the screen if requested
    if verbose:
        for attribute, value in data.items():
            print(attribute, ':', value)
        print('-' * 75)

    return data

class Demand():
    """
    Description of an engineering demand parameter.

    Attributes
    ----------
    ID: string
        Explain...
    kind: {'PFA', 'PID', 'RD'}
        Explain...
    cline: int
        Explain...
    floor: int
        Explain...
    floor2: int, optional
        default: None
        Explain..
    raw_data: DataFrame
        Explain... rows=models, columns=events

    """

    def __init__(self, kind, cline, floor, RV, floor2=None):
        self._kind = kind
        self._cline = cline
        self._floor = floor
        self._RV = RV
        self._floor2 = floor2

    @property
    def ID(self):
        return '{}_{}'.format(self._kind, self._floor)