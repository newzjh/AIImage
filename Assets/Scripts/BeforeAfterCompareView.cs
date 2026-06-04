using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

public sealed class BeforeAfterCompareView : VisualElement
{
    private readonly Label _beforeTag;
    private readonly Label _afterTag;
    private readonly Label _leftHint;
    private readonly Label _rightHint;

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
    public float AngleRad { get; private set; }

    public BeforeAfterCompareView()
    {
        style.flexGrow = 1;
        style.minHeight = 0;
        style.overflow = Overflow.Hidden;
        style.backgroundColor = new StyleColor(new Color(0.10f, 0.11f, 0.13f, 1f));

        _beforeTag = CreateTag("Before");
        _beforeTag.style.left = 18;
        _beforeTag.style.top = 18;
        Add(_beforeTag);

        _afterTag = CreateTag("After");
        _afterTag.style.right = 18;
        _afterTag.style.top = 18;
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
        MarkDirtyRepaint();
        UpdateDecorations();
    }

    public void SetPreview(Texture preview)
    {
        _previewTex = preview;
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
        MarkDirtyRepaint();
        UpdateDecorations();
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
        MarkDirtyRepaint();
        UpdateDecorations();
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
        if (!InteractionEnabled)
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
        MarkDirtyRepaint();
        UpdateDecorations();
        evt.StopPropagation();
    }

    private void OnPointerDown(PointerDownEvent evt)
    {
        if (!InteractionEnabled)
            return;

        var refTex = _texA ?? _texB;
        if (refTex == null)
            return;

        if (evt.clickCount == 2)
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
            UpdateDecorations();
            evt.StopPropagation();
            return;
        }

        if (!_panning || _panPointerId != evt.pointerId || !this.HasPointerCapture(_panPointerId))
            return;

        _pan = _panStartPan + (Vector2)(evt.localPosition - _panStartPointer);
        MarkDirtyRepaint();
        UpdateDecorations();
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

    private Rect GetViewRect()
    {
        return contentRect;
    }

    private static Rect GetImageRect(Texture refTex, float zoom, Vector2 pan)
    {
        return new Rect(pan.x, pan.y, refTex.width * zoom, refTex.height * zoom);
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
        if (drawRect.width <= 1f || drawRect.height <= 1f)
        {
            UpdateDecorations();
            return;
        }

        ClampOffsetToDrawRect(drawRect, imageRect);

        if (_previewTex != null)
        {
            DrawFullRect(mgc, _previewTex, drawRect, imageRect);
            PositionHints(imageRect, drawRect);
            return;
        }

        float SignedDistance(Vector2 point) => SignedDistUv(point, imageRect);
        if (_texA != null)
            DrawHalfPlane(mgc, _texA, drawRect, imageRect, SignedDistance, true);
        if (_texB != null)
            DrawHalfPlane(mgc, _texB, drawRect, imageRect, SignedDistance, false);
        DrawSplitLine(mgc, drawRect, imageRect);
        PositionHints(imageRect, drawRect);
    }

    private void PositionHints(Rect imageRect, Rect drawRect)
    {
        if (!TryGetSplitSegment(drawRect, imageRect, out var segA, out var segB))
        {
            _leftHint.style.display = DisplayStyle.None;
            _rightHint.style.display = DisplayStyle.None;
            return;
        }

        var mid = (segA + segB) * 0.5f;
        var y = Mathf.Clamp(mid.y, imageRect.yMin + 28f, imageRect.yMax - 28f);
        _leftHint.style.left = mid.x - 28f;
        _leftHint.style.top = y - 14f;
        _rightHint.style.left = mid.x + 6f;
        _rightHint.style.top = y - 14f;
        _leftHint.style.display = DisplayStyle.Flex;
        _rightHint.style.display = DisplayStyle.Flex;
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

    private static void DrawFullRect(MeshGenerationContext mgc, Texture tex, Rect drawRect, Rect imageRect)
    {
        var mesh = mgc.Allocate(4, 6, tex);
        var p0 = new Vector2(drawRect.xMin, drawRect.yMin);
        var p1 = new Vector2(drawRect.xMax, drawRect.yMin);
        var p2 = new Vector2(drawRect.xMax, drawRect.yMax);
        var p3 = new Vector2(drawRect.xMin, drawRect.yMax);

        mesh.SetNextVertex(new Vertex { position = p0, uv = Uv(p0, imageRect), tint = Color.white });
        mesh.SetNextVertex(new Vertex { position = p1, uv = Uv(p1, imageRect), tint = Color.white });
        mesh.SetNextVertex(new Vertex { position = p2, uv = Uv(p2, imageRect), tint = Color.white });
        mesh.SetNextVertex(new Vertex { position = p3, uv = Uv(p3, imageRect), tint = Color.white });
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
        label.style.backgroundColor = Color.white;
        label.style.color = new Color(0.62f, 0.62f, 0.64f, 1f);
        label.style.unityFontStyleAndWeight = FontStyle.Bold;
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
        label.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.22f));
        label.style.borderTopLeftRadius = 11;
        label.style.borderTopRightRadius = 11;
        label.style.borderBottomLeftRadius = 11;
        label.style.borderBottomRightRadius = 11;
        label.style.display = DisplayStyle.None;
        return label;
    }
}
