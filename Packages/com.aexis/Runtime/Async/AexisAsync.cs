using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Aexis.Async
{
    /// <summary>Centralizes the engine's frame scheduling policy without third-party async dependencies.</summary>
    public static class AexisAsync
    {
        /// <summary>Completes after Unity advances to the next frame.</summary>
        public static async Task YieldFrame()
        {
#if UNITY_2023_1_OR_NEWER
            await Awaitable.NextFrameAsync();
#else
            await Unity2022FrameYield.YieldAsync();
#endif
        }

#if !UNITY_2023_1_OR_NEWER
        // Awaitable.NextFrameAsync was introduced after Unity 2022.  Task.Yield only
        // posts to UnitySynchronizationContext and can run again in the same player
        // loop iteration, so 2022 keeps a small main-thread driver that completes a
        // request only after Time.frameCount has advanced.
        private static class Unity2022FrameYield
        {
            private sealed class Waiter
            {
                public int requestedFrame;
                public TaskCompletionSource<bool> completion;
            }

            private sealed class Driver : MonoBehaviour
            {
                private void Awake()
                {
                    hideFlags = HideFlags.HideAndDontSave;
                    DontDestroyOnLoad(gameObject);
                }

                private void Update()
                {
                    CompleteEligibleWaiters(Time.frameCount);
                }

                private void OnDestroy()
                {
                    lock (Gate)
                    {
                        if (ReferenceEquals(_driver, this))
                            _driver = null;
                    }
                }
            }

            private static readonly object Gate = new object();
            private static readonly List<Waiter> Waiters = new List<Waiter>();
            private static Driver _driver;
            private static int _mainThreadId;

            [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
            private static void CaptureMainThread()
            {
                _mainThreadId = Thread.CurrentThread.ManagedThreadId;
            }

            public static Task YieldAsync()
            {
                // Edit-mode and worker-thread callers do not have a player-loop driver.
                // They retain the old cooperative behavior instead of touching Unity
                // objects from a non-main thread.
                if (!Application.isPlaying
                    || _mainThreadId == 0
                    || Thread.CurrentThread.ManagedThreadId != _mainThreadId)
                {
                    return YieldCooperativelyAsync();
                }

                var waiter = new Waiter
                {
                    requestedFrame = Time.frameCount,
                    completion = new TaskCompletionSource<bool>()
                };
                lock (Gate)
                {
                    EnsureDriver();
                    Waiters.Add(waiter);
                }
                return waiter.completion.Task;
            }

            private static async Task YieldCooperativelyAsync()
            {
                await Task.Yield();
            }

            private static void EnsureDriver()
            {
                if (_driver != null)
                    return;

                var gameObject = new GameObject("AexisAsync.Unity2022FrameYield")
                {
                    hideFlags = HideFlags.HideAndDontSave
                };
                _driver = gameObject.AddComponent<Driver>();
            }

            private static void CompleteEligibleWaiters(int currentFrame)
            {
                List<Waiter> completed = null;
                lock (Gate)
                {
                    for (var index = Waiters.Count - 1; index >= 0; index--)
                    {
                        var waiter = Waiters[index];
                        if (waiter.requestedFrame >= currentFrame)
                            continue;

                        completed ??= new List<Waiter>();
                        completed.Add(waiter);
                        Waiters.RemoveAt(index);
                    }
                }

                if (completed == null)
                    return;
                for (var index = 0; index < completed.Count; index++)
                    completed[index].completion.TrySetResult(true);
            }
        }
#endif
    }
}
