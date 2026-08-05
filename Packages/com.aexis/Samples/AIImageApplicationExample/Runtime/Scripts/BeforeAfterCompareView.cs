using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

public sealed class BeforeAfterCompareView : VisualElement
{
    public event Action ViewTransformChanged;

    private readonly Label _beforeTag;
    private readonly Label _afterTag;
    private readonly Label _leftHint;
    private readonly Label _rightHint;
    private Rect _lastImageRect;
    private Rect _lastDrawRect;
    private bool _hasPendingDecorationLayout;

    private Texture _texA;
    private Texture _texB;
    private Texture _previewTex;
    private float _zoom = 1f;
    private Vector2 _pan;
    private bool _panning;
    private int _panPointerId;
    private Vector3 _panStartPointer;
    private Vector2 _panStartPan;
    private bool _dragSplit;
    private int _splitPointerId;
    private Vector3 _splitDragStartLocal;
    private float _splitDragStartAngle;
    private float _splitDragStartOffset;
    private float _offset;
    private const float ThicknessPx = 2f;
    private static readonly Color LineColor = new Color(1f, 1f, 1f, 0.95f);

    public bool InteractionEnabled { get; set; } = true;
    public bool LocalMaskPaintingEnabled { get; set; }
    public float LocalMaskBrushSize { get; set; } = 0.08f;
    public Texture LocalMaskOverlay { get; set; }
    public float AngleRad { get; private set; }
    public event Action<Vector2, float, bool> LocalMaskStroke;
    private bool _paintingLocalMask;
    private int _localMaskPointerId = -1;

    public BeforeAfterCompareView()
    {
        style.flexGrow = 1;
        style.minHeight = 0;
        style.overflow = Overflow.Hidden;
        style.backgroundColor = new StyleColor(new Color(0.10f, 0.11f, 0.13f, 1f));

        _beforeTag = CreateTag("Before");
        Add(_beforeTag);

        _afterTag = CreateTag("After");
        Add(_afterTag);

        _leftHint = CreateHint("<");
        Add(_leftHint);

        _rightHint = CreateHint(">");
        Add(_rightHint);

        generateVisualContent += OnGenerateVisualContent;
        RegisterCallback<GeometryChangedEvent>(_ =>
        {
            if (_texA != null || _texB != null)
                FitToView();
            UpdateDecorations();
        });
        RegisterCallback<WheelEvent>(OnWheel, TrickleDown.TrickleDown);
        RegisterCallback<PointerDownEvent>(OnPointerDown);
        RegisterCallback<PointerMoveEvent>(OnPointerMove);
        RegisterCallback<PointerUpEvent>(OnPointerUp);
        RegisterCallback<PointerCancelEvent>(OnPointerCancel);
    }

    public void SetSources(Texture current, Texture original, string _)
    {
        _texA = current;
        _texB = original;
        _hasPendingDecorationLayout = true;
        MarkDirtyRepaint();
        UpdateDecorations();
    }

    public void SetPreview(Texture preview)
    {
        _previewTex = preview;
        _hasPendingDecorationLayout = true;
        MarkDirtyRepaint();
    }

    public new void Clear()
    {
        _texA = null;
        _texB = null;
        _previewTex = null;
        ResetView();
        MarkDirtyRepaint();
        UpdateDecorations();
    }

    public void ResetView()
    {
        _zoom = 1f;
        var refTex = _texA ?? _texB;
        if (refTex != null)
        {
            var viewRect = GetViewRect();
            var viewportSize = viewRect.size;
            var scaledSize = new Vector2(refTex.width * _zoom, refTex.height * _zoom);
            _pan = (viewportSize - scaledSize) * 0.5f;
        }
        else
        {
            _pan = Vector2.zero;
        }
        _hasPendingDecorationLayout = true;
        MarkDirtyRepaint();
        UpdateDecorations();
        NotifyViewTransformChanged();
    }

    public void FitToView()
    {
        var refTex = _texA ?? _texB;
        if (refTex == null)
            return;

        var viewRect = GetViewRect();
        if (viewRect.width <= 1f || viewRect.height <= 1f)
            return;

        var scaleX = viewRect.width / refTex.width;
        var scaleY = viewRect.height / refTex.height;
        _zoom = Mathf.Clamp(Mathf.Min(scaleX, scaleY), 0.01f, 20f);
        var scaledSize = new Vector2(refTex.width * _zoom, refTex.height * _zoom);
        _pan = (viewRect.size - scaledSize) * 0.5f;
        _hasPendingDecorationLayout = true;
        MarkDirtyRepaint();
        UpdateDecorations();
        NotifyViewTransformChanged();
    }

    public bool TryGetDisplayedImageRect(out Rect imageRect)
    {
        var refTex = _texA ?? _texB;
        if (refTex == null)
        {
            imageRect = default;
            return false;
        }
        var viewRect = GetViewRect();
        imageRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
        return imageRect.width > 1f && imageRect.height > 1f;
    }

    private void OnWheel(WheelEvent evt)
    {
        if (!InteractionEnabled && !LocalMaskPaintingEnabled)
            return;
        var refTex = _texA ?? _texB;
        var viewRect = GetViewRect();
        if (refTex == null || viewRect.width <= 1f || viewRect.height <= 1f || !viewRect.Contains(evt.localMousePosition))
            return;

        var imageRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
        var drawRect = IntersectRect(viewRect, imageRect);
        if (!drawRect.Contains(evt.localMousePosition))
            return;

        var viewportPos = evt.localMousePosition - viewRect.position;
        var oldZoom = _zoom;
        var factor = Mathf.Pow(1.12f, -evt.delta.y / 12f);
        _zoom = Mathf.Clamp(oldZoom * factor, 0.02f, 40f);
        if (Mathf.Approximately(oldZoom, _zoom))
            return;

        var imageLocal = (viewportPos - _pan) / oldZoom;
        _pan = viewportPos - imageLocal * _zoom;
        _hasPendingDecorationLayout = true;
        MarkDirtyRepaint();
        UpdateDecorations();
        NotifyViewTransformChanged();
        evt.StopPropagation();
    }

    private void OnPointerDown(PointerDownEvent evt)
    {
        if (!InteractionEnabled)
            return;

        var refTex = _texA ?? _texB;
        if (refTex == null)
            return;

        if (evt.clickCount == 2 && !LocalMaskPaintingEnabled)
        {
            FitToView();
            evt.StopPropagation();
            return;
        }

        if (evt.button != 0 && evt.button != 2)
            return;

        var viewRect = GetViewRect();
        var imageRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
        var drawRect = IntersectRect(viewRect, imageRect);
        if (!drawRect.Contains(evt.localPosition))
            return;

        if (LocalMaskPaintingEnabled && evt.button == 0)
        {
            _paintingLocalMask = true;
            _localMaskPointerId = evt.pointerId;
            EmitLocalMaskPoint(evt.localPosition, imageRect, true);
            this.CapturePointer(_localMaskPointerId);
            evt.StopPropagation();
            return;
        }

        if (evt.button == 0)
        {
            var signedDistance = SignedDistUv(evt.localPosition, imageRect);
            var thresholdUv = 12f / Mathf.Min(imageRect.width, imageRect.height);
            if (Mathf.Abs(signedDistance) <= thresholdUv)
            {
                _dragSplit = true;
                _splitPointerId = evt.pointerId;
                _splitDragStartLocal = evt.localPosition;
                _splitDragStartAngle = AngleRad;
                _splitDragStartOffset = _offset;
                this.CapturePointer(_splitPointerId);
                evt.StopPropagation();
                return;
            }
        }

        _panning = true;
        _panPointerId = evt.pointerId;
        _panStartPointer = evt.localPosition;
        _panStartPan = _pan;
        this.CapturePointer(_panPointerId);
        evt.StopPropagation();
    }

    private void OnPointerMove(PointerMoveEvent evt)
    {
        var refTex = _texA ?? _texB;
        if (refTex == null)
            return;

        if (_paintingLocalMask && _localMaskPointerId == evt.pointerId && this.HasPointerCapture(_localMaskPointerId))
        {
            var viewRect = GetViewRect();
            var imageRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
            if (IntersectRect(viewRect, imageRect).Contains(evt.localPosition))
                EmitLocalMaskPoint(evt.localPosition, imageRect, false);
            evt.StopPropagation();
            return;
        }

        if (_dragSplit && _splitPointerId == evt.pointerId && this.HasPointerCapture(_splitPointerId))
        {
            var viewRect = GetViewRect();
            var imageRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
            var drawRect = IntersectRect(viewRect, imageRect);
            var deltaLocal = evt.localPosition - _splitDragStartLocal;
            var deltaUv = new Vector2(deltaLocal.x / Mathf.Max(1f, imageRect.width), -deltaLocal.y / Mathf.Max(1f, imageRect.height));

            if (evt.shiftKey)
            {
                var deltaAngle = (deltaLocal.x / Mathf.Max(1f, imageRect.width)) * Mathf.PI * 2f;
                AngleRad = _splitDragStartAngle + deltaAngle;
                ClampOffsetToDrawRect(drawRect, imageRect);
            }
            else
            {
                var normal = new Vector2(Mathf.Cos(AngleRad), Mathf.Sin(AngleRad));
                _offset = _splitDragStartOffset - Vector2.Dot(normal, deltaUv);
                ClampOffsetToDrawRect(drawRect, imageRect);
            }
            MarkDirtyRepaint();
            _hasPendingDecorationLayout = true;
            UpdateDecorations();
            NotifyViewTransformChanged();
            evt.StopPropagation();
            return;
        }

        if (!_panning || _panPointerId != evt.pointerId || !this.HasPointerCapture(_panPointerId))
            return;

        _pan = _panStartPan + (Vector2)(evt.localPosition - _panStartPointer);
        _hasPendingDecorationLayout = true;
        MarkDirtyRepaint();
        UpdateDecorations();
        NotifyViewTransformChanged();
        evt.StopPropagation();
    }

    private void OnPointerUp(PointerUpEvent evt)
    {
        if (_paintingLocalMask && _localMaskPointerId == evt.pointerId)
        {
            _paintingLocalMask = false;
            if (this.HasPointerCapture(_localMaskPointerId))
                this.ReleasePointer(_localMaskPointerId);
            _localMaskPointerId = -1;
            evt.StopPropagation();
            return;
        }

        if (_dragSplit && _splitPointerId == evt.pointerId)
        {
            _dragSplit = false;
            if (this.HasPointerCapture(_splitPointerId))
                this.ReleasePointer(_splitPointerId);
            evt.StopPropagation();
            return;
        }

        if (!_panning || _panPointerId != evt.pointerId)
            return;
        _panning = false;
        if (this.HasPointerCapture(_panPointerId))
            this.ReleasePointer(_panPointerId);
        evt.StopPropagation();
    }

    private void OnPointerCancel(PointerCancelEvent evt)
    {
        if (_paintingLocalMask && _localMaskPointerId == evt.pointerId)
        {
            _paintingLocalMask = false;
            if (this.HasPointerCapture(_localMaskPointerId))
                this.ReleasePointer(_localMaskPointerId);
            _localMaskPointerId = -1;
            evt.StopPropagation();
            return;
        }

        if (_dragSplit && _splitPointerId == evt.pointerId)
        {
            _dragSplit = false;
            if (this.HasPointerCapture(_splitPointerId))
                this.ReleasePointer(_splitPointerId);
            evt.StopPropagation();
            return;
        }

        if (_panning && _panPointerId == evt.pointerId)
        {
            _panning = false;
            if (this.HasPointerCapture(_panPointerId))
                this.ReleasePointer(_panPointerId);
            evt.StopPropagation();
        }
    }

    private Rect GetViewRect()
    {
        return contentRect;
    }

    private static Rect GetImageRect(Texture refTex, float zoom, Vector2 pan)
    {
        return new Rect(pan.x, pan.y, refTex.width * zoom, refTex.height * zoom);
    }

    private void EmitLocalMaskPoint(Vector2 localPosition, Rect imageRect, bool strokeStart)
    {
        var uv = new Vector2(
            Mathf.Clamp01((localPosition.x - imageRect.xMin) / Mathf.Max(1f, imageRect.width)),
            Mathf.Clamp01(1f - (localPosition.y - imageRect.yMin) / Mathf.Max(1f, imageRect.height)));
        LocalMaskStroke?.Invoke(uv, Mathf.Clamp(LocalMaskBrushSize, 0.005f, 1f), strokeStart);
    }

    private float SignedDistUv(Vector2 pointLocal, Rect imageRect)
    {
        var normal = new Vector2(Mathf.Cos(AngleRad), Mathf.Sin(AngleRad));
        var uv = new Vector2(
            (pointLocal.x - imageRect.xMin) / imageRect.width,
            1f - ((pointLocal.y - imageRect.yMin) / imageRect.height));
        return Vector2.Dot(normal, uv - new Vector2(0.5f, 0.5f)) + _offset;
    }

    private void ClampOffsetToDrawRect(Rect drawRect, Rect imageRect)
    {
        if (drawRect.width <= 1f || drawRect.height <= 1f || imageRect.width <= 1f || imageRect.height <= 1f)
            return;

        var normal = new Vector2(Mathf.Cos(AngleRad), Mathf.Sin(AngleRad));
        if (normal.sqrMagnitude <= 1e-6f)
            return;
        normal.Normalize();

        Vector2 UvOf(Vector2 point)
        {
            return new Vector2(
                (point.x - imageRect.xMin) / imageRect.width,
                1f - ((point.y - imageRect.yMin) / imageRect.height));
        }

        var corners = new[]
        {
            new Vector2(drawRect.xMin, drawRect.yMin),
            new Vector2(drawRect.xMax, drawRect.yMin),
            new Vector2(drawRect.xMax, drawRect.yMax),
            new Vector2(drawRect.xMin, drawRect.yMax)
        };

        var center = new Vector2(0.5f, 0.5f);
        float D(Vector2 uv) => Vector2.Dot(normal, uv - center);
        var distances = new float[corners.Length];
        for (var i = 0; i < corners.Length; i++)
            distances[i] = D(UvOf(corners[i]));
        var minDistance = distances.Min();
        var maxDistance = distances.Max();
        var minOffset = -maxDistance;
        var maxOffset = -minDistance;
        _offset = Mathf.Clamp(_offset, minOffset, maxOffset);
    }

    private void OnGenerateVisualContent(MeshGenerationContext mgc)
    {
        var refTex = _texA ?? _texB;
        if (refTex == null)
        {
            UpdateDecorations();
            return;
        }

        var viewRect = GetViewRect();
        if (viewRect.width <= 1f || viewRect.height <= 1f)
        {
            UpdateDecorations();
            return;
        }

        var imageRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
        var drawRect = IntersectRect(viewRect, imageRect);
        _lastImageRect = imageRect;
        _lastDrawRect = drawRect;
        if (drawRect.width <= 1f || drawRect.height <= 1f)
        {
            UpdateDecorations();
            return;
        }

        ClampOffsetToDrawRect(drawRect, imageRect);

        if (_previewTex != null)
        {
            DrawFullRect(mgc, _previewTex, drawRect, imageRect);
            DrawLocalMaskOverlay(mgc, drawRect, imageRect);
            ScheduleDecorationLayout();
            return;
        }

        float SignedDistance(Vector2 point) => SignedDistUv(point, imageRect);
        if (_texA != null)
            DrawHalfPlane(mgc, _texA, drawRect, imageRect, SignedDistance, true);
        if (_texB != null)
            DrawHalfPlane(mgc, _texB, drawRect, imageRect, SignedDistance, false);
        DrawSplitLine(mgc, drawRect, imageRect);
        DrawLocalMaskOverlay(mgc, drawRect, imageRect);
        ScheduleDecorationLayout();
    }

    private void ScheduleDecorationLayout()
    {
        if (!_hasPendingDecorationLayout)
            return;

        _hasPendingDecorationLayout = false;
        schedule.Execute(UpdateDecorationLayout);
    }

    private void UpdateDecorationLayout()
    {
        var refTex = _texA ?? _texB;
        if (refTex == null)
            return;

        var imageRect = _lastImageRect;
        var drawRect = _lastDrawRect;
        if (imageRect.width <= 1f || imageRect.height <= 1f || drawRect.width <= 1f || drawRect.height <= 1f)
        {
            _leftHint.style.display = DisplayStyle.None;
            _rightHint.style.display = DisplayStyle.None;
            return;
        }

        PositionHints(imageRect, drawRect);
    }

    private void PositionHints(Rect imageRect, Rect drawRect)
    {
        if (!TryGetSplitSegment(drawRect, imageRect, out var segA, out var segB))
        {
            PositionTags(imageRect, false, 0f, 0f);
            _leftHint.style.display = DisplayStyle.None;
            _rightHint.style.display = DisplayStyle.None;
            return;
        }

        var mid = (segA + segB) * 0.5f;
        var tangent = (segB - segA).normalized;
        if (tangent.sqrMagnitude <= 1e-6f)
            tangent = Vector2.right;

        var normal = new Vector2(Mathf.Cos(AngleRad), -Mathf.Sin(AngleRad));
        if (normal.sqrMagnitude <= 1e-6f)
            normal = Vector2.up;
        else
            normal.Normalize();

        var rotationDeg = -AngleRad * Mathf.Rad2Deg;
        PositionTags(imageRect, true, mid.x, mid.y, normal, rotationDeg);
        PositionHint(_leftHint, mid - normal * 24f, rotationDeg);
        PositionHint(_rightHint, mid + normal * 24f, rotationDeg);
        _leftHint.style.display = DisplayStyle.Flex;
        _rightHint.style.display = DisplayStyle.Flex;
    }

    private void PositionTags(Rect imageRect, bool nearSplit, float splitX, float splitY, Vector2 splitNormal = default, float rotationDeg = 0f)
    {
        if (!nearSplit)
        {
            _afterTag.style.left = imageRect.xMin + 18f;
            _afterTag.style.top = imageRect.yMin + 18f;
            _beforeTag.style.left = Mathf.Max(imageRect.xMin + 18f, imageRect.xMax - 90f);
            _beforeTag.style.top = imageRect.yMin + 18f;
            _afterTag.style.rotate = new Rotate(new Angle(0f, AngleUnit.Degree));
            _beforeTag.style.rotate = new Rotate(new Angle(0f, AngleUnit.Degree));
            _leftHint.style.rotate = new Rotate(new Angle(0f, AngleUnit.Degree));
            _rightHint.style.rotate = new Rotate(new Angle(0f, AngleUnit.Degree));
            return;
        }

        if (splitNormal.sqrMagnitude <= 1e-6f)
            splitNormal = Vector2.up;
        else
            splitNormal.Normalize();

        PositionTag(_afterTag, new Vector2(splitX, splitY) - splitNormal * 64f, rotationDeg, imageRect);
        PositionTag(_beforeTag, new Vector2(splitX, splitY) + splitNormal * 84f, rotationDeg, imageRect);
    }

    private static void PositionTag(VisualElement tag, Vector2 center, float rotationDeg, Rect imageRect)
    {
        var width = Mathf.Max(82f, tag.resolvedStyle.width > 1f ? tag.resolvedStyle.width : 86f);
        var height = Mathf.Max(28f, tag.resolvedStyle.height > 1f ? tag.resolvedStyle.height : 34f);
        var left = Mathf.Clamp(center.x - width * 0.5f, imageRect.xMin + 6f, imageRect.xMax - width - 6f);
        var top = Mathf.Clamp(center.y - height * 0.5f, imageRect.yMin + 6f, imageRect.yMax - height - 6f);
        tag.style.left = left;
        tag.style.top = top;
        tag.style.rotate = new Rotate(new Angle(rotationDeg, AngleUnit.Degree));
        tag.style.transformOrigin = new TransformOrigin(Length.Percent(50), Length.Percent(50), 0f);
    }

    private static void PositionHint(VisualElement hint, Vector2 center, float rotationDeg)
    {
        var width = Mathf.Max(22f, hint.resolvedStyle.width > 1f ? hint.resolvedStyle.width : 22f);
        var height = Mathf.Max(22f, hint.resolvedStyle.height > 1f ? hint.resolvedStyle.height : 22f);
        hint.style.left = center.x - width * 0.5f;
        hint.style.top = center.y - height * 0.5f;
        hint.style.rotate = new Rotate(new Angle(rotationDeg, AngleUnit.Degree));
        hint.style.transformOrigin = new TransformOrigin(Length.Percent(50), Length.Percent(50), 0f);
    }

    private void UpdateDecorations()
    {
        var hasImage = (_texA ?? _texB) != null;
        _beforeTag.style.display = hasImage ? DisplayStyle.Flex : DisplayStyle.None;
        _afterTag.style.display = hasImage ? DisplayStyle.Flex : DisplayStyle.None;
        if (!hasImage)
        {
            _leftHint.style.display = DisplayStyle.None;
            _rightHint.style.display = DisplayStyle.None;
        }
    }

    private void NotifyViewTransformChanged()
    {
        ViewTransformChanged?.Invoke();
    }

    private void DrawLocalMaskOverlay(MeshGenerationContext mgc, Rect drawRect, Rect imageRect)
    {
        if (!LocalMaskPaintingEnabled || LocalMaskOverlay == null)
            return;

        DrawFullRect(mgc, LocalMaskOverlay, drawRect, imageRect, new Color(0.10f, 0.70f, 1.00f, 0.32f));
    }

    private static void DrawFullRect(MeshGenerationContext mgc, Texture tex, Rect drawRect, Rect imageRect, Color? tintOverride = null)
    {
        var mesh = mgc.Allocate(4, 6, tex);
        var p0 = new Vector2(drawRect.xMin, drawRect.yMin);
        var p1 = new Vector2(drawRect.xMax, drawRect.yMin);
        var p2 = new Vector2(drawRect.xMax, drawRect.yMax);
        var p3 = new Vector2(drawRect.xMin, drawRect.yMax);

        var tint = tintOverride ?? Color.white;
        mesh.SetNextVertex(new Vertex { position = p0, uv = Uv(p0, imageRect), tint = tint });
        mesh.SetNextVertex(new Vertex { position = p1, uv = Uv(p1, imageRect), tint = tint });
        mesh.SetNextVertex(new Vertex { position = p2, uv = Uv(p2, imageRect), tint = tint });
        mesh.SetNextVertex(new Vertex { position = p3, uv = Uv(p3, imageRect), tint = tint });
        mesh.SetNextIndex(0);
        mesh.SetNextIndex(1);
        mesh.SetNextIndex(2);
        mesh.SetNextIndex(0);
        mesh.SetNextIndex(2);
        mesh.SetNextIndex(3);
    }

    private static void DrawHalfPlane(MeshGenerationContext mgc, Texture tex, Rect drawRect, Rect imageRect, Func<Vector2, float> signedDistFunc, bool keepNegative)
    {
        var rectPoly = new List<Vector2>
        {
            new Vector2(drawRect.xMin, drawRect.yMin),
            new Vector2(drawRect.xMax, drawRect.yMin),
            new Vector2(drawRect.xMax, drawRect.yMax),
            new Vector2(drawRect.xMin, drawRect.yMax)
        };

        var clipped = ClipPolygon(rectPoly, signedDistFunc, keepNegative);
        if (clipped.Count < 3)
            return;

        var mesh = mgc.Allocate(clipped.Count, (clipped.Count - 2) * 3, tex);
        for (var i = 0; i < clipped.Count; i++)
            mesh.SetNextVertex(new Vertex { position = clipped[i], uv = Uv(clipped[i], imageRect), tint = Color.white });
        for (var i = 0; i < clipped.Count - 2; i++)
        {
            mesh.SetNextIndex(0);
            mesh.SetNextIndex((ushort)(i + 1));
            mesh.SetNextIndex((ushort)(i + 2));
        }
    }

    private void DrawSplitLine(MeshGenerationContext mgc, Rect drawRect, Rect imageRect)
    {
        if (!TryGetSplitSegment(drawRect, imageRect, out var segA, out var segB))
            return;

        var dir = (segB - segA).normalized;
        if (dir.sqrMagnitude <= 1e-6f)
            return;
        var perp = new Vector2(-dir.y, dir.x) * (ThicknessPx * 0.5f);
        var quad = new List<Vector2>
        {
            segA + perp,
            segA - perp,
            segB - perp,
            segB + perp
        };
        var clipped = ClipToRect(quad, drawRect);
        if (clipped.Count < 3)
            return;

        var mesh = mgc.Allocate(clipped.Count, (clipped.Count - 2) * 3, Texture2D.whiteTexture);
        for (var i = 0; i < clipped.Count; i++)
            mesh.SetNextVertex(new Vertex { position = clipped[i], uv = Vector2.zero, tint = LineColor });
        for (var i = 0; i < clipped.Count - 2; i++)
        {
            mesh.SetNextIndex(0);
            mesh.SetNextIndex((ushort)(i + 1));
            mesh.SetNextIndex((ushort)(i + 2));
        }
    }

    private bool TryGetSplitSegment(Rect drawRect, Rect imageRect, out Vector2 segA, out Vector2 segB)
    {
        segA = default;
        segB = default;
        if (imageRect.width <= 1f || imageRect.height <= 1f)
            return false;

        var normal = new Vector2(Mathf.Cos(AngleRad), Mathf.Sin(AngleRad));
        var width = imageRect.width;
        var height = imageRect.height;
        var x0 = imageRect.xMin;
        var y1 = imageRect.yMax;
        var c = 0.5f * (normal.x + normal.y) - _offset;
        var a = normal.x / width;
        var b = -normal.y / height;
        var d = (-normal.x * x0 / width) + (normal.y * y1 / height) - c;
        float F(Vector2 point) => a * point.x + b * point.y + d;

        var corners = new[]
        {
            new Vector2(drawRect.xMin, drawRect.yMin),
            new Vector2(drawRect.xMax, drawRect.yMin),
            new Vector2(drawRect.xMax, drawRect.yMax),
            new Vector2(drawRect.xMin, drawRect.yMax)
        };

        var points = new List<Vector2>(4);
        for (var i = 0; i < 4; i++)
        {
            var p0 = corners[i];
            var p1 = corners[(i + 1) % 4];
            var f0 = F(p0);
            var f1 = F(p1);

            if (Mathf.Abs(f0) <= 1e-6f)
                points.Add(p0);
            if ((f0 > 0f && f1 < 0f) || (f0 < 0f && f1 > 0f))
            {
                var t = f0 / (f0 - f1);
                points.Add(p0 + (p1 - p0) * t);
            }
        }

        if (points.Count < 2)
            return false;

        segA = points[0];
        segB = points[1];
        var bestDist = (segB - segA).sqrMagnitude;
        for (var i = 0; i < points.Count; i++)
        {
            for (var j = i + 1; j < points.Count; j++)
            {
                var distance = (points[j] - points[i]).sqrMagnitude;
                if (distance > bestDist)
                {
                    bestDist = distance;
                    segA = points[i];
                    segB = points[j];
                }
            }
        }
        return true;
    }

    private static Rect IntersectRect(Rect a, Rect b)
    {
        var xMin = Mathf.Max(a.xMin, b.xMin);
        var yMin = Mathf.Max(a.yMin, b.yMin);
        var xMax = Mathf.Min(a.xMax, b.xMax);
        var yMax = Mathf.Min(a.yMax, b.yMax);
        if (xMax <= xMin || yMax <= yMin)
            return new Rect(0, 0, 0, 0);
        return Rect.MinMaxRect(xMin, yMin, xMax, yMax);
    }

    private static List<Vector2> ClipToRect(List<Vector2> polygon, Rect rect)
    {
        var p = polygon;
        p = ClipPolygon(p, v => rect.xMin - v.x, true);
        if (p.Count < 3) return p;
        p = ClipPolygon(p, v => v.x - rect.xMax, true);
        if (p.Count < 3) return p;
        p = ClipPolygon(p, v => rect.yMin - v.y, true);
        if (p.Count < 3) return p;
        p = ClipPolygon(p, v => v.y - rect.yMax, true);
        return p;
    }

    private static List<Vector2> ClipPolygon(List<Vector2> polygon, Func<Vector2, float> signedDistance, bool keepNegative)
    {
        var output = new List<Vector2>(polygon.Count + 4);
        if (polygon.Count == 0)
            return output;

        var previous = polygon[^1];
        var previousDistance = signedDistance(previous);
        var previousInside = keepNegative ? previousDistance <= 0f : previousDistance >= 0f;

        for (var i = 0; i < polygon.Count; i++)
        {
            var current = polygon[i];
            var currentDistance = signedDistance(current);
            var currentInside = keepNegative ? currentDistance <= 0f : currentDistance >= 0f;

            if (currentInside)
            {
                if (!previousInside)
                    output.Add(Intersect(previous, current, previousDistance, currentDistance));
                output.Add(current);
            }
            else if (previousInside)
            {
                output.Add(Intersect(previous, current, previousDistance, currentDistance));
            }

            previous = current;
            previousDistance = currentDistance;
            previousInside = currentInside;
        }

        return output;
    }

    private static Vector2 Intersect(Vector2 a, Vector2 b, float da, float db)
    {
        var t = da / (da - db);
        return a + (b - a) * t;
    }

    private static Vector2 Uv(Vector2 point, Rect imageRect)
    {
        return new Vector2(
            (point.x - imageRect.xMin) / imageRect.width,
            1f - ((point.y - imageRect.yMin) / imageRect.height));
    }

    private static Label CreateTag(string text)
    {
        var label = new Label(text);
        label.style.position = Position.Absolute;
        label.style.paddingLeft = 12;
        label.style.paddingRight = 12;
        label.style.paddingTop = 8;
        label.style.paddingBottom = 8;
        label.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.52f));
        label.style.color = Color.white;
        label.style.unityFontStyleAndWeight = FontStyle.Bold;
        label.style.fontSize = 13;
        label.style.borderLeftWidth = 1;
        label.style.borderRightWidth = 1;
        label.style.borderTopWidth = 1;
        label.style.borderBottomWidth = 1;
        label.style.borderLeftColor = new StyleColor(new Color(1f, 1f, 1f, 0.20f));
        label.style.borderRightColor = new StyleColor(new Color(1f, 1f, 1f, 0.20f));
        label.style.borderTopColor = new StyleColor(new Color(1f, 1f, 1f, 0.20f));
        label.style.borderBottomColor = new StyleColor(new Color(1f, 1f, 1f, 0.20f));
        label.style.borderTopLeftRadius = 18;
        label.style.borderTopRightRadius = 18;
        label.style.borderBottomLeftRadius = 18;
        label.style.borderBottomRightRadius = 18;
        return label;
    }

    private static Label CreateHint(string text)
    {
        var label = new Label(text);
        label.style.position = Position.Absolute;
        label.style.width = 22;
        label.style.height = 22;
        label.style.unityTextAlign = TextAnchor.MiddleCenter;
        label.style.color = Color.white;
        label.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.48f));
        label.style.borderTopLeftRadius = 11;
        label.style.borderTopRightRadius = 11;
        label.style.borderBottomLeftRadius = 11;
        label.style.borderBottomRightRadius = 11;
        label.style.display = DisplayStyle.None;
        return label;
    }
}
