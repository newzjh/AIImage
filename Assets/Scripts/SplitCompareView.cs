using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

/// <summary>
/// 完整的分割对比视图 - 从MainView.cs提取并修复UV翻转
/// 支持Before/After对比、分割线、拖拽、缩放等功能
/// </summary>
public class SplitCompareView : VisualElement
{
    private readonly Label _info;
    private readonly Label _beforeLabel;
    private readonly Label _afterLabel;

    private Texture _texA;
    private Texture _texB;
    private Texture _previewTex;
    public float angleRad;
    private float offset;
    private float thicknessPx = 2f;
    private Color lineColor = new Color(1f, 1f, 1f, 0.9f);

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

    public SplitCompareView()
    {
        style.flexDirection = FlexDirection.Column;
        style.flexGrow = 1;
        style.overflow = Overflow.Hidden;

        _info = new Label();
        _info.style.flexShrink = 0;
        _info.style.paddingLeft = 8;
        _info.style.paddingRight = 8;
        _info.style.paddingTop = 6;
        _info.style.paddingBottom = 6;
        _info.style.whiteSpace = WhiteSpace.NoWrap;
        _info.style.overflow = Overflow.Hidden;
        _info.style.textOverflow = TextOverflow.Ellipsis;
        _info.style.color = Color.white;
        Add(_info);

        style.flexGrow = 1;
        style.minHeight = 0;
        style.backgroundColor = new StyleColor(new Color(0.12f, 0.12f, 0.12f, 1f));

        pickingMode = PickingMode.Position;
        angleRad = 0f;
        offset = 0f;

        // Before/After标签
        _beforeLabel = new Label("BEFORE");
        _beforeLabel.style.position = Position.Absolute;
        _beforeLabel.style.left = 20;
        _beforeLabel.style.top = 20;
        _beforeLabel.style.color = new Color(1f, 1f, 1f, 0.8f);
        _beforeLabel.style.fontSize = 16;
        _beforeLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
        Add(_beforeLabel);

        _afterLabel = new Label("AFTER");
        _afterLabel.style.position = Position.Absolute;
        _afterLabel.style.right = 20;
        _afterLabel.style.top = 20;
        _afterLabel.style.color = new Color(1f, 1f, 1f, 0.8f);
        _afterLabel.style.fontSize = 16;
        _afterLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
        Add(_afterLabel);

        generateVisualContent += OnGenerateVisualContent;
        RegisterCallback<GeometryChangedEvent>(_ => { if (_texA != null || _texB != null) FitToView(); });
        RegisterCallback<WheelEvent>(OnWheel, TrickleDown.TrickleDown);
        RegisterCallback<PointerDownEvent>(OnPointerDown);
        RegisterCallback<PointerMoveEvent>(OnPointerMove);
        RegisterCallback<PointerUpEvent>(OnPointerUp);
        RegisterCallback<PointerCancelEvent>(OnPointerCancel);
    }

    public void SetSources(Texture2D current, Texture2D original, string label)
    {
        _texA = current;
        _texB = original;

        var refTex = _texA != null ? _texA : _texB;
        _info.text = refTex == null ? "" : (label ?? refTex.name);
        MarkDirtyRepaint();
    }

    public void SetPreview(Texture preview)
    {
        _previewTex = preview;
        MarkDirtyRepaint();
    }

    public void Clear()
    {
        SetSources(null, null, null);
        ResetView();
    }

    public void ResetView()
    {
        _zoom = 1f;
        var refTex = _texA != null ? _texA : _texB;
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
        MarkDirtyRepaint();
    }

    public void FitToView()
    {
        var refTex = _texA != null ? _texA : _texB;
        if (refTex == null) return;

        var viewRect = GetViewRect();
        var viewportSize = viewRect.size;
        if (viewportSize.x <= 1f || viewportSize.y <= 1f) return;

        var scaleX = viewportSize.x / refTex.width;
        var scaleY = viewportSize.y / refTex.height;
        _zoom = Mathf.Clamp(Mathf.Min(scaleX, scaleY), 0.01f, 20f);

        var scaledSize = new Vector2(refTex.width * _zoom, refTex.height * _zoom);
        _pan = (viewportSize - scaledSize) * 0.5f;
        MarkDirtyRepaint();
    }

    private void OnWheel(WheelEvent evt)
    {
        if (_texA == null && _texB == null) return;

        var refTex0 = _texA != null ? _texA : _texB;
        var viewRect0 = GetViewRect();
        if (viewRect0.width <= 1f || viewRect0.height <= 1f)
            return;
        if (!viewRect0.Contains(evt.localMousePosition))
            return;

        var imgRect0 = GetImageRect(refTex0, _zoom, viewRect0.position + _pan);
        var drawRect0 = IntersectRect(viewRect0, imgRect0);
        if (!drawRect0.Contains(evt.localMousePosition))
            return;

        var viewportPos = evt.localMousePosition - viewRect0.position;
        var oldZoom = _zoom;
        var factor = Mathf.Pow(1.12f, -evt.delta.y / 12f);
        _zoom = Mathf.Clamp(oldZoom * factor, 0.02f, 40f);

        if (Mathf.Approximately(oldZoom, _zoom)) return;

        var imageLocal = (viewportPos - _pan) / oldZoom;
        _pan = viewportPos - imageLocal * _zoom;
        MarkDirtyRepaint();

        evt.StopPropagation();
    }

    private void OnPointerDown(PointerDownEvent evt)
    {
        if (_texA == null && _texB == null) return;

        if (evt.clickCount == 2)
        {
            FitToView();
            evt.StopPropagation();
            return;
        }

        if (evt.button != 0 && evt.button != 2) return;

        var refTex = _texA != null ? _texA : _texB;
        var viewRect = GetViewRect();
        if (viewRect.width <= 1f || viewRect.height <= 1f)
            return;
        if (!viewRect.Contains(evt.localPosition))
            return;

        var imgRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
        if (imgRect.width <= 1f || imgRect.height <= 1f)
            return;
        var drawRect = IntersectRect(viewRect, imgRect);
        if (!drawRect.Contains(evt.localPosition))
            return;

        if (evt.button == 0)
        {
            var sd = SignedDistUv(evt.localPosition, imgRect);
            var thresholdUv = 12f / Mathf.Min(imgRect.width, imgRect.height);
            if (Mathf.Abs(sd) <= thresholdUv)
            {
                _dragSplit = true;
                _splitPointerId = evt.pointerId;
                _splitDragStartLocal = evt.localPosition;
                _splitDragStartAngle = angleRad;
                _splitDragStartOffset = offset;
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
        if (_dragSplit && _splitPointerId == evt.pointerId && this.HasPointerCapture(_splitPointerId))
        {
            var refTex = _texA != null ? _texA : _texB;
            var viewRect = GetViewRect();
            var imgRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
            var w = Mathf.Max(1f, imgRect.width);
            var h = Mathf.Max(1f, imgRect.height);
            var deltaLocal = evt.localPosition - _splitDragStartLocal;
            var deltaUv = new Vector2(deltaLocal.x / w, -deltaLocal.y / h);
            var drawRect = IntersectRect(viewRect, imgRect);

            if (evt.shiftKey)
            {
                var deltaAngle = (deltaLocal.x / w) * Mathf.PI * 2f;
                angleRad = _splitDragStartAngle + deltaAngle;
                ClampOffsetToDrawRect(drawRect, imgRect);
            }
            else
            {
                var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
                offset = _splitDragStartOffset - Vector2.Dot(n, deltaUv);
                ClampOffsetToDrawRect(drawRect, imgRect);
            }

            MarkDirtyRepaint();
            evt.StopPropagation();
            return;
        }

        if (!_panning || _panPointerId != evt.pointerId || !this.HasPointerCapture(_panPointerId))
            return;

        _pan = _panStartPan + (Vector2)(evt.localPosition - _panStartPointer);
        MarkDirtyRepaint();
        evt.StopPropagation();
    }

    private void OnPointerUp(PointerUpEvent evt)
    {
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

    private static Rect GetImageRect(Texture refTex, float zoom, Vector2 pan)
    {
        if (refTex == null) return default;
        return new Rect(pan.x, pan.y, refTex.width * zoom, refTex.height * zoom);
    }

    private Rect GetViewRect()
    {
        var bounds = contentRect;
        if (bounds.width <= 1f || bounds.height <= 1f)
            return new Rect(0, 0, 0, 0);

        var infoBottom = _info.layout.y + _info.layout.height;
        infoBottom = Mathf.Clamp(infoBottom, bounds.yMin, bounds.yMax);
        return Rect.MinMaxRect(bounds.xMin, infoBottom, bounds.xMax, bounds.yMax);
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

    private float SignedDistUv(Vector2 pLocal, Rect imageRect)
    {
        var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
        var uv = new Vector2(
            (pLocal.x - imageRect.xMin) / imageRect.width,
            1f - ((pLocal.y - imageRect.yMin) / imageRect.height)
        );
        return Vector2.Dot(n, uv - new Vector2(0.5f, 0.5f)) + offset;
    }

    private void ClampOffsetToDrawRect(Rect drawRect, Rect imageRect)
    {
        if (drawRect.width <= 1f || drawRect.height <= 1f)
            return;
        if (imageRect.width <= 1f || imageRect.height <= 1f)
            return;

        var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
        if (n.sqrMagnitude <= 1e-6f)
            return;
        n.Normalize();

        Vector2 UvOf(Vector2 pLocal)
        {
            return new Vector2(
                (pLocal.x - imageRect.xMin) / imageRect.width,
                1f - ((pLocal.y - imageRect.yMin) / imageRect.height)
            );
        }

        var p0 = new Vector2(drawRect.xMin, drawRect.yMin);
        var p1 = new Vector2(drawRect.xMax, drawRect.yMin);
        var p2 = new Vector2(drawRect.xMax, drawRect.yMax);
        var p3 = new Vector2(drawRect.xMin, drawRect.yMax);

        var c = new Vector2(0.5f, 0.5f);
        float D(Vector2 uv) => Vector2.Dot(n, uv - c);

        var d0 = D(UvOf(p0));
        var d1 = D(UvOf(p1));
        var d2 = D(UvOf(p2));
        var d3 = D(UvOf(p3));

        var minD = Mathf.Min(Mathf.Min(d0, d1), Mathf.Min(d2, d3));
        var maxD = Mathf.Max(Mathf.Max(d0, d1), Mathf.Max(d2, d3));

        var minOffset = -maxD;
        var maxOffset = -minD;
        if (minOffset > maxOffset)
        {
            var t = minOffset;
            minOffset = maxOffset;
            maxOffset = t;
        }

        offset = Mathf.Clamp(offset, minOffset, maxOffset);
    }

    private void OnGenerateVisualContent(MeshGenerationContext mgc)
    {
        var refTex = _texA != null ? _texA : _texB;
        if (refTex == null)
            return;

        var viewRect = GetViewRect();
        if (viewRect.width <= 1f || viewRect.height <= 1f)
            return;

        var imageRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
        var drawRect = IntersectRect(viewRect, imageRect);
        if (drawRect.width <= 1f || drawRect.height <= 1f)
            return;

        ClampOffsetToDrawRect(drawRect, imageRect);

        if (_previewTex != null)
        {
            DrawFullRect(mgc, _previewTex, drawRect, imageRect);
            return;
        }
        float SignedDist(Vector2 p) => SignedDistUv(p, imageRect);

        if (_texA != null)
            DrawHalfPlane(mgc, _texA, drawRect, imageRect, SignedDist, keepNegative: true);
        if (_texB != null)
            DrawHalfPlane(mgc, _texB, drawRect, imageRect, SignedDist, keepNegative: false);

        DrawSplitLine(mgc, drawRect, imageRect);
        DrawArrows(mgc, drawRect, imageRect);
    }

    private static void DrawFullRect(MeshGenerationContext mgc, Texture tex, Rect drawRect, Rect imageRect)
    {
        var mesh = mgc.Allocate(4, 6, tex);
        var p0 = new Vector2(drawRect.xMin, drawRect.yMin);
        var p1 = new Vector2(drawRect.xMax, drawRect.yMin);
        var p2 = new Vector2(drawRect.xMax, drawRect.yMax);
        var p3 = new Vector2(drawRect.xMin, drawRect.yMax);

        Vector2 Uv(Vector2 p)
        {
            return new Vector2(
                (p.x - imageRect.xMin) / imageRect.width,
                1f - ((p.y - imageRect.yMin) / imageRect.height)  // 修复UV翻转
            );
        }

        mesh.SetNextVertex(new Vertex { position = p0, uv = Uv(p0), tint = Color.white });
        mesh.SetNextVertex(new Vertex { position = p1, uv = Uv(p1), tint = Color.white });
        mesh.SetNextVertex(new Vertex { position = p2, uv = Uv(p2), tint = Color.white });
        mesh.SetNextVertex(new Vertex { position = p3, uv = Uv(p3), tint = Color.white });

        mesh.SetNextIndex(0);
        mesh.SetNextIndex(1);
        mesh.SetNextIndex(2);
        mesh.SetNextIndex(0);
        mesh.SetNextIndex(2);
        mesh.SetNextIndex(3);
    }

    private void DrawHalfPlane(
        MeshGenerationContext mgc,
        Texture tex,
        Rect drawRect,
        Rect imageRect,
        Func<Vector2, float> signedDistFunc,
        bool keepNegative)
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

        var vCount = clipped.Count;
        var iCount = (vCount - 2) * 3;
        var mesh = mgc.Allocate(vCount, iCount, tex);

        for (int i = 0; i < vCount; i++)
        {
            var p = clipped[i];
            var uv = new Vector2(
                (p.x - imageRect.xMin) / imageRect.width,
                1f - ((p.y - imageRect.yMin) / imageRect.height)  // 修复UV翻转
            );
            mesh.SetNextVertex(new Vertex
            {
                position = p,
                uv = uv,
                tint = Color.white
            });
        }

        for (int i = 0; i < vCount - 2; i++)
        {
            mesh.SetNextIndex(0);
            mesh.SetNextIndex((ushort)(i + 1));
            mesh.SetNextIndex((ushort)(i + 2));
        }
    }

    private void DrawSplitLine(MeshGenerationContext mgc, Rect drawRect, Rect imageRect)
    {
        if (imageRect.width <= 1f || imageRect.height <= 1f)
            return;

        var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
        var w = imageRect.width;
        var h = imageRect.height;
        var x0 = imageRect.xMin;
        var y0 = imageRect.yMin;
        var y1 = imageRect.yMax;
        var c = 0.5f * (n.x + n.y) - offset;
        var A = n.x / w;
        var B = -n.y / h;
        var C = (-n.x * x0 / w) + (n.y * y1 / h) - c;

        float F(Vector2 p) => A * p.x + B * p.y + C;

        var corners = new[]
        {
            new Vector2(drawRect.xMin, drawRect.yMin),
            new Vector2(drawRect.xMax, drawRect.yMin),
            new Vector2(drawRect.xMax, drawRect.yMax),
            new Vector2(drawRect.xMin, drawRect.yMax)
        };

        var points = new List<Vector2>(4);
        for (int i = 0; i < 4; i++)
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
            return;

        var segA = points[0];
        var segB = points[1];
        float bestDist = (segB - segA).sqrMagnitude;
        for (int i = 0; i < points.Count; i++)
        {
            for (int j = i + 1; j < points.Count; j++)
            {
                var d = (points[j] - points[i]).sqrMagnitude;
                if (d > bestDist)
                {
                    bestDist = d;
                    segA = points[i];
                    segB = points[j];
                }
            }
        }

        var dir = segB - segA;
        if (dir.sqrMagnitude < 1e-4f)
            return;
        dir.Normalize();
        var perp = new Vector2(-dir.y, dir.x) * (thicknessPx * 0.5f);

        var v0p = segA + perp;
        var v1p = segA - perp;
        var v2p = segB - perp;
        var v3p = segB + perp;

        var quad = new List<Vector2> { v0p, v1p, v2p, v3p };
        var clipped = ClipToRect(quad, drawRect);
        if (clipped.Count < 3)
            return;

        var vCount = clipped.Count;
        var iCount = (vCount - 2) * 3;
        var mesh = mgc.Allocate(vCount, iCount, Texture2D.whiteTexture);
        for (int i = 0; i < vCount; i++)
            mesh.SetNextVertex(new Vertex { position = clipped[i], uv = Vector2.zero, tint = lineColor });
        for (int i = 0; i < vCount - 2; i++)
        {
            mesh.SetNextIndex(0);
            mesh.SetNextIndex((ushort)(i + 1));
            mesh.SetNextIndex((ushort)(i + 2));
        }
    }

    // 绘制箭头（<和>符号）
    private void DrawArrows(MeshGenerationContext mgc, Rect drawRect, Rect imageRect)
    {
        if (imageRect.width <= 1f || imageRect.height <= 1f)
            return;

        // 计算分割线的中心点
        var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
        var centerX = imageRect.center.x;
        var centerY = imageRect.center.y;

        // 简化：在分割线两侧绘制箭头提示
        // 注意：这里简化实现，实际应该根据分割线位置计算
    }

    private static List<Vector2> ClipToRect(List<Vector2> poly, Rect rect)
    {
        List<Vector2> p = poly;
        p = ClipPolygon(p, v => rect.xMin - v.x, true);
        if (p.Count < 3) return p;
        p = ClipPolygon(p, v => v.x - rect.xMax, true);
        if (p.Count < 3) return p;
        p = ClipPolygon(p, v => rect.yMin - v.y, true);
        if (p.Count < 3) return p;
        p = ClipPolygon(p, v => v.y - rect.yMax, true);
        return p;
    }

    private static List<Vector2> ClipPolygon(List<Vector2> poly, Func<Vector2, float> signedDist, bool keepNegative)
    {
        var output = new List<Vector2>(poly.Count + 4);
        if (poly.Count == 0)
            return output;

        Vector2 prev = poly[poly.Count - 1];
        float prevD = signedDist(prev);
        bool prevIn = keepNegative ? prevD <= 0f : prevD >= 0f;

        for (int i = 0; i < poly.Count; i++)
        {
            Vector2 cur = poly[i];
            float curD = signedDist(cur);
            bool curIn = keepNegative ? curD <= 0f : curD >= 0f;

            if (curIn)
            {
                if (!prevIn)
                {
                    output.Add(Intersect(prev, cur, prevD, curD));
                }
                output.Add(cur);
            }
            else if (prevIn)
            {
                output.Add(Intersect(prev, cur, prevD, curD));
            }

            prev = cur;
            prevD = curD;
            prevIn = curIn;
        }

        return output;
    }

    private static Vector2 Intersect(Vector2 a, Vector2 b, float da, float db)
    {
        var t = da / (da - db);
        return a + (b - a) * t;
    }
}
