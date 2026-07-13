using System.Runtime.CompilerServices;

// AIImage application runners were formerly compiled beside this runtime.
// Keep their existing internal integration surface during the staged UPM extraction.
[assembly: InternalsVisibleTo("Assembly-CSharp")]
[assembly: InternalsVisibleTo("Assembly-CSharp-Editor")]
