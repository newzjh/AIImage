# Sample Dependency Notice

This directory is part of the application example only. It is not referenced by Aexis Runtime.

- `AexisSampleAsync` is a namespace-isolated source copy derived from the UniTask runtime required by the original AIImage application. Its public namespaces are rewritten from `Cysharp.Threading.Tasks` to `Aexis.Samples.Async` and it compiles as `Aexis.Sample.Async`.
- `AexisSampleSharpZip` is a namespace-isolated source copy derived from the SharpZipLib code required by the original AIImage application. Its public namespaces are rewritten from `ICSharpCode.SharpZipLib` to `Aexis.Samples.SharpZipLib`.

The rewrite prevents assembly and namespace collisions with a consuming project's own UniTask or SharpZipLib distribution. It does not alter upstream licenses. Keep the copyright and license headers in the copied source files. Before publishing a package archive, record each source's upstream URL, immutable revision, complete license text, modifications, and redistribution approval in the release record.
