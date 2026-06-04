# MainView2 / LibraryView / DesignView 本次对话修改总结

日期：2026-06-04

## 目标

本次对话主要围绕新的 3 个页面 `MainView2`、`LibraryView`、`DesignView` 的细节修复、交互补齐、异步加载优化和稳定性处理展开，要求同时满足：

- `AIImage.sln` / `csproj` 可编译通过
- Unity 静默模式可正常启动
- 不修改原有 `MainView.cs`

## 本次实际修改的文件

- `Assets/Scripts/BasePageView.cs`
- `Assets/Scripts/BeforeAfterCompareView.cs`
- `Assets/Scripts/LibraryView.cs`
- `Assets/Scripts/MainView2.cs`
- `Assets/Scripts/DesignView.cs`

## 已完成的主要修改

### 1. LibraryView 缩略图与目录侧

- 将缩略图卡片放大到原先约 `1.5x`，并保持自动换行。
- 修复首批缩略图一直显示“加载中”的问题。
- 缩略图刷新改为后台逐步处理：
  - 目录扫描放到后台线程
  - 文件读取放到后台线程
  - EXIF 提取放到后台线程
  - 缩略图缩放放到后台线程
  - Unity `Texture2D` 落地仍在主线程
- 页面切换、目录切换时会中断已有缩略图后台刷新。
- 增加图片元数据提取：
  - 拍摄时间
  - GPS 位置（当前先显示坐标）
  - 相机型号
  - 光圈
- `LibraryView` 返回时会恢复选中缩略图，并自动滚动到该缩略图位置。
- `LibraryView` 首次打开时，会按 `MainView2` 当前图片路径做一次性同步：
  - 展开对应目录
  - 选中对应缩略图
  - 自动滚动定位
  - 仅首次生效，之后继续沿用 `LibraryView` 自己的状态逻辑
- 左侧目录树增加更接近中文/自然顺序的目录排序。
- 文件名排序改为尽量对齐 Windows 资源管理器逻辑：
  - 使用 `StrCmpLogicalW`
  - 默认“名称”排序按文件名顺序
  - 人脸/地点排序时文件名作为次级排序

### 2. MainView2 页面

- 修复 `Fit` 后图片没有正确占满可用区域的问题。
- 右侧浮动调节面板支持拖动。
- 历史面板支持拖动。
- 带分割线的图片区域不再覆盖顶部按钮栏。
- 底部预设按钮条改为覆盖在图片区域之上，更接近 `mainview2.jpg` 参考图层级。
- 底部页面切换滑块改为覆盖在图片区域之上。

### 3. DesignView 页面

- 历史面板支持拖动。
- 说明/提示面板支持拖动。
- 修复分割线无法拖动的问题：
  - 根因是上层 `_canvasOverlay` 拦截了 CompareView 的指针事件
  - 现已改为不阻断分割线拖拽
- 图层框仍可继续拖动与缩放。
- 底部页面切换滑块改为覆盖在图片区域之上。

### 4. Before / After 对比分割线视图

- 增加更明显的 `Before` / `After` 文案。
- 增加分割线两侧的 `<` / `>` 提示。
- 文案位置从固定角落改为贴近分割线两侧显示。
- 修复旋转分割线后 4 个字样未跟随的问题。
- 修复旋转角度偏差 90 度的问题。
- 修复由于在 `generateVisualContent` 回调中修改样式导致的异常：
  - 原异常：
    `InvalidOperationException: VisualElements cannot change their render data under an active visual tree during generateVisualContent callback execution...`
  - 处理方式：
    - 绘制阶段只记录当前分割线和图像区域几何
    - 标签位置与旋转改为在后续调度中更新

### 5. 基础页面能力 BasePageView

- 新增通用浮动面板拖动能力 `EnableFloatingPanelDrag(...)`。
- Toast 飘字整体下移 `100px`，避免挡住第一排按钮。
- 底部页面切换滑块支持按页面状态固定停靠：
  - `LibraryView` 最左
  - `MainView2` 中间
  - `DesignView` 最右
- 支持底部滑块以覆盖层方式显示，而不是始终占用文档流空间。

## 本次明确修复过的问题清单

- `LibraryView` 某些目录缩略图加载不完整
- `LibraryView` 图片多时刷新卡顿
- `LibraryView` 页面切换时后台刷新不能中断
- `LibraryView` 缩略图需要放大
- `LibraryView` 切走再回来丢失选中状态
- `LibraryView` 返回时不自动滚动到已选中缩略图
- `LibraryView` 首次打开不能按 `MainView2` 当前图定位
- `LibraryView` 默认排序不符合资源管理器名称顺序
- `MainView2` / `DesignView` 缺少 `Before` / `After`
- `MainView2` `Fit` 后没有真正占满区域
- `MainView2` 右侧面板和历史面板标题栏无法拖动
- `DesignView` 分割线拖不动
- `DesignView` 图层框交互与分割线事件冲突
- Toast 飘字挡住顶部第一排按钮
- `MainView2` / `DesignView` 图片区域与底部条层级不符合参考图
- 底部页面滑块没有位于最左 / 中 / 最右正确位置
- 分割线旋转后 `Before` / `After` / `<` / `>` 不跟随
- 分割线旋转后字样角度偏差 90 度
- 旋转标签更新触发 UI Toolkit 渲染阶段异常

## 关键实现说明

### 1. LibraryView 的异步处理

- 目录扫描通过 `UniTask.RunOnThreadPool(...)` 放后台。
- 单张缩略图读取与 EXIF 解析也放后台。
- UI 侧每张图逐步刷新，并在切页/切目录时通过 `CancellationTokenSource` 中断。
- 缩略图处理加入 `DelayFrame(1)`，减少长时间占用主线程连续刷新。

### 2. 文件名排序

- 当前文件名比较器使用 Windows `shlwapi.dll` 的 `StrCmpLogicalW`。
- 这比普通字符串比较更接近资源管理器的“按名称”规则。
- 非 Windows 情况下回退到 `CompareInfo("zh-CN")` 的字符串比较。

### 3. 分割线标签更新

- 不能在 `generateVisualContent` 中直接改 `VisualElement.style`。
- 当前做法是：
  - 绘制时更新 `_lastImageRect` / `_lastDrawRect`
  - 通过 `schedule.Execute(...)` 延后执行 `UpdateDecorationLayout()`
  - 在调度回调里更新 `Before/After/< />` 的位置、旋转和显隐

## 验证结果

本次修改过程中多次执行以下验证，最终状态均通过：

### .NET 编译

```powershell
dotnet build AIImage.sln -nologo -clp:ErrorsOnly
```

结果：

- `0 errors`

### Unity 静默启动

```powershell
"C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe" -batchmode -nographics -quit -projectPath "E:\Projects\AIImage"
```

结果：

- 退出码 `0`
- 可以正常静默启动并退出

## 建议后续回归重点

### 1. LibraryView

- 用一组典型文件名验证资源管理器顺序是否完全一致：
  - `DSCF0415.jpg`
  - `DSCF0415a.jpg`
  - `DSCF0416.jpg`
  - `10-1安阳殷墟`
  - `10-3郑州开封`
- 验证首次进入 `LibraryView` 时是否一定按 `MainView2` 当前图正确展开和定位。
- 验证切换目录、切换页面时后台缩略图任务是否都能及时中断。

### 2. Before / After

- 旋转分割线时检查：
  - `Before`
  - `After`
  - `<`
  - `>`
  是否都能跟着旋转且保持在分割线两侧。

### 3. 底部覆盖层级

- 对照 `ref/layout/mainview2.jpg`
- 重点确认：
  - `MainView2` 底排按钮压在图片区域上方
  - `MainView2` 底部滑块压在图片区域上方
  - `DesignView` 底部滑块压在图片区域上方

## 当前仍可继续优化的方向

- GPS 坐标可继续接入“中文地名反解”。
- `LibraryView` 的名称排序如果后续仍发现与资源管理器个别边界情况不一致，可以继续对比较器做更细节拟合。
- `LibraryView` 未来还可继续接入 CLIP 分类、人脸、地点等真实数据源，而不是当前预留字段。

