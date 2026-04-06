from skyfield.api import load

def download_tle():
    max_days = 7.0         # download again once 7 days old
    group = 'starlink'
    name = f'{group}.tle'

    base = 'https://celestrak.org/NORAD/elements/gp.php'
    url = base + f'?GROUP={group}&FORMAT=tle'

    if not load.exists(name) or load.days_old(name) >= max_days:
        load.download(url, filename=name)

if __name__ == '__main__':
    # download_tle() # first time only
    pass
