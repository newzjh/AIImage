# Aexis Sample Json Source Record

This directory is a namespace-isolated source copy of Json.NET used only by the
AIImage Main2 application example. It is not referenced by any Aexis Runtime
assembly and it does not ship a `Newtonsoft.Json.dll`.

- Upstream: https://github.com/JamesNK/Newtonsoft.Json
- Source revision: Json.NET 13.0.2, commit `4fba53a324c445f06ee08e45a015c346000a7ef2`
- Source archive: https://codeload.github.com/JamesNK/Newtonsoft.Json/zip/refs/tags/13.0.2
- Archive SHA-256: `3FE7BBF46E2745E7742ECFA6A18E276999512597B017FB4AC5ED86655FF2FCBA`
- License: MIT, reproduced in `LICENSE.md`
- Redistribution: permitted by the upstream MIT license with the copyright and
  license notice retained.

Local modifications are limited to mechanical namespace shading from
`Newtonsoft.Json` to `Aexis.Samples.Json`, preserving the source layout needed
by the application sample, and per-file source capability symbols matching the
upstream `netstandard2.0` build profile. These changes prevent a consuming
project's own Json.NET package or DLL from colliding with this sample-only copy.
