#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

using System;
using System.Collections;

namespace Aexis.Samples.Async
{
    // UnityEngine Bridges.

    public partial struct UniTask
    {
        public static IEnumerator ToCoroutine(Func<UniTask> taskFactory)
        {
            return taskFactory().ToCoroutine();
        }
    }
}

