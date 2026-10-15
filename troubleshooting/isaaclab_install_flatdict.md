# Isaac Lab 설치 후 `No module named 'isaaclab'` — 원인은 flatdict 빌드 실패

> 환경: Ubuntu 24.04 / RTX 5070 Ti / Isaac Sim 5.1 / Isaac Lab 2.3.2 / conda 환경 `isaaclab`

---

## 1. 증상

`./isaaclab.sh --install rsl_rl`이 끝난 뒤 예제를 실행하면:

```text
ModuleNotFoundError: No module named 'isaaclab'
```

- `torch.cuda.is_available()`은 `True`, CUDA 12.8로 GPU 쪽은 정상이었다.
- 설치 명령은 에러 없이 끝난 것처럼 보였다.

## 2. 원인 분석

### 2.1 무엇이 없는지 확인

```bash
pip list | grep -i isaaclab
```

`isaaclab_tasks` 같은 **하위 패키지는 설치돼 있는데, 본체 `isaaclab`만 없었다.**
→ 본체 설치 단계 어딘가에서 실패했다는 뜻.

### 2.2 설치 로그를 거슬러 올라가기

설치 로그에서 `ERROR`를 검색하니, 본체의 의존성인 `flatdict==4.0.1`을 빌드하다 실패한 기록이 있었다.

```text
No module named 'pkg_resources'
```

### 2.3 근본 원인

- pip는 소스 패키지를 빌드할 때 **build isolation**(격리된 임시 환경)을 쓰고, 그 안에 최신 setuptools를 새로 설치한다.
- 그 최신 setuptools에는 `pkg_resources`가 없어서 `flatdict` 빌드가 실패했다.
- 그런데 나머지 패키지 설치는 계속 진행돼서, 전체 명령은 **정상 종료처럼 보였다.**

```text
직접 원인:  import isaaclab 실패
    ↑
중간 원인:  isaaclab 본체 미설치
    ↑
근본 원인:  flatdict 빌드 실패 (build isolation 안의 최신 setuptools에 pkg_resources 없음)
```

## 3. 해결

```bash
# isaaclab conda 환경에서
pip install "setuptools<81" wheel
pip install flatdict==4.0.1 --no-build-isolation   # 격리 없이, 방금 설치한 setuptools로 빌드
./isaaclab.sh --install rsl_rl
```

해결 후 확인:

```bash
pip list | grep -i -E "^(setuptools|flatdict|isaaclab) "
# setuptools 80.10.2 / flatdict 4.0.1 / isaaclab 0.54.2
python -c "import isaaclab; print('ok')"
```

## 4. 배운 점

- **설치 명령이 끝났다고 성공한 게 아니다.** 여러 패키지를 한 번에 설치하면 하나가 실패해도 나머지는 계속 진행된다.
  설치 후 `pip list`로 실제로 깔렸는지 확인하고, 로그에서 `ERROR`를 검색한다.
- **직접 원인과 근본 원인은 다를 수 있다.** 에러 메시지(`No module named 'isaaclab'`)만 보고 경로나 환경 문제로 추측하지 말고,
  로그를 거슬러 올라가 처음 실패한 지점을 찾아야 한다.
- 다음부터는 설치 로그를 파일로 남긴다:

  ```bash
  ./isaaclab.sh --install rsl_rl 2>&1 | tee install.log
  grep -n -i error install.log
  ```
