
# UnityEduGameScaffold (Unity 2022.3 LTS friendly)

**这是一个“能跑起来的空项目脚手架（Assets 级别）”**，用于竞赛教育类（仿真+教学+评分）的 Unity 工程快速起步：

- 无需任何预制体/场景资源，按下 **Play** 即可：
  - 自动生成 Canvas + EventSystem + 主菜单；
  - “开始模拟”后自动构建一个简单关卡、生成一辆差速小车和 HUD；
  - 键盘 **W/S** 前后、**A/D** 转向（老 InputManager 轴），HUD 显示速度/距离传感器/得分。
- 代码组织已分层：Core / Simulation / UI / Programming / Scoring，便于后续扩展。

> 目标是：**零依赖、零场景、零资源** 也能跑，避免你因为包/版本/URP 等环境差异卡住。

---

## 快速使用

1. 在 Unity Hub 中创建 **Unity 2022.3 LTS (推荐 URP 模板或 3D 模板均可)** 的新项目；
2. 将本压缩包中的 `Assets/` 覆盖复制到你项目的 `Assets/` 目录（允许合并）；
3. 打开项目，保持默认 SampleScene 或任意空场景，**直接按 Play**；
4. 点击“开始模拟”，使用键盘控制小车，吃到小球得分。

> 若使用 URP 模板：脚本与功能均不受影响；只是渲染更优。

---

## 目录说明

```
Assets/
  Scripts/
    Core/            # Bootstrap/GameApp/EventBus（运行时自动拉起 + 应用壳）
    Simulation/      # 机器人、差速控制器、传感器、关卡搭建器
      Sensors/
      Environment/
    UI/              # 主菜单 + HUD
    Programming/     # IProgramVM 接口 + 简单占位 VM
    Scoring/         # 计分器（拾取物加分），事件总线广播
  README.md
```

- **Bootstrap.cs**：无需场景预设，Play 时自动生成 `GameApp`；
- **GameApp.cs**：创建 UI Canvas、主菜单，点击“开始模拟”动态搭关卡、生成 `Robot` 与 HUD；
- **Robot.cs + DifferentialDriveController.cs**：差速底盘运动学（基于 Rigidbody.MovePosition/Rotation）；
- **DistanceSensor.cs**：简单前向测距（Raycast）；
- **LevelBuilder.cs**：50x50 地面 + 围墙 + 随机 5 个拾取点（触发即加分）；
- **ScoreManager.cs**：计分单例 + 事件；
- **MainMenuUI.cs / HUD.cs**：完全运行时创建的 UGUI，避免资源依赖；
- **IProgramVM.cs / SimpleVM.cs**：编程 VM 规范与占位实现（后续可接 uBlockly/语法糖 → IR → VM）。

---

## 已做的框架约束（便于后续扩展）

- **确定性雏形**：物理更新走 `FixedUpdate`，后续可加入固定步长的 VM `Step(dt)` 与随机种子；
- **UI 运行时生成**：UGUI 在纯代码中创建，避免 Prefab/Font/TMP 依赖；
- **无新包依赖**：仅使用内置 `UnityEngine.UI`；避免 Input System 依赖导致编译错误。

---

## 下一步建议

- **加入 Addressables**：把课程/关卡/题面做成可热更新包；
- **编程形态**：接入 uBlockly / NodeGraph，统一编译到 VM 指令集；
- **更多传感器**：循迹（贴图采样）、IMU（积分 + 噪声）、多束雷达；
- **赛事评分**：规则 DSL（ScriptableObject），录像/回放仲裁；
- **真机 I/O**：抽象 `IRobotIO`，串口 / BLE / WebSocket 适配器。

---

## 常见问题

- **启动报错缺少 EventSystem/Canvas？**  
  本脚手架会自动创建；若你已有同名对象，脚手架会复用或在新场景再次生成。

- **Input System 包未安装会不会错误？**  
  不会。本脚手架使用旧版 `Input.GetAxis`。若你引入 Input System，请自行切换输入层。

- **为什么没有 .unity 场景文件？**  
  为避免跨版本序列化差异，场景在运行时动态生成。构建前，你可以手动保存当前层级为场景并加入 Build Settings。
