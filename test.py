from pettingzoo.test import parallel_api_test
from SatelliteDataDisseminationEnv import SatelliteDataDisseminationEnv


def main():
    env = SatelliteDataDisseminationEnv()
    # 這行會自動用隨機動作幫你跑過幾百個 step，檢查有沒有任何格式、維度不合的 bug
    parallel_api_test(env, num_cycles=1000)
    print("環境測試完美通過！可以開始訓練了！")

if __name__ == '__main__':
    main()