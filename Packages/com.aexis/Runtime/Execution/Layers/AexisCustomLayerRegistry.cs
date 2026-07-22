using System;
using System.Collections.Generic;

namespace Aexis.Execution
{
    [Serializable]
    public sealed class AexisLayerParameterSchema
    {
        // Either ncnnKey or name identifies a parameter. ncnnKey is ignored when
        // hasNcnnKey is false, which keeps named ONNX/custom parameters unambiguous.
        public bool hasNcnnKey;
        public int ncnnKey;
        public string name = string.Empty;
        public bool required;
        public bool allowEmpty;

        public void Validate(AexisGraphModel.Layer layer, string layerType)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            var hasValue = hasNcnnKey
                ? layer.intParams != null && layer.intParams.TryGetValue(ncnnKey, out var value) && (allowEmpty || !string.IsNullOrWhiteSpace(value))
                : !string.IsNullOrWhiteSpace(name)
                    && layer.stringParams != null
                    && layer.stringParams.TryGetValue(name, out var namedValue)
                    && (allowEmpty || !string.IsNullOrWhiteSpace(namedValue));
            if (required && !hasValue)
            {
                var parameter = hasNcnnKey ? ncnnKey.ToString() : name ?? string.Empty;
                throw new InvalidOperationException(
                    "Custom layer parameter is required"
                    + " | type=" + (layerType ?? string.Empty)
                    + " | layer=" + (layer.name ?? string.Empty)
                    + " | parameter=" + parameter);
            }
        }
    }

    [Serializable]
    public sealed class AexisCustomLayerSchema
    {
        public int schemaVersion = 1;
        public int minimumInputs;
        public int maximumInputs = -1;
        public int minimumOutputs = 1;
        public int maximumOutputs = -1;
        public bool textureNativeRequired = true;
        public AexisLayerParameterSchema[] parameters = Array.Empty<AexisLayerParameterSchema>();

        public void Validate(AexisGraphModel.Layer layer, string layerType)
        {
            if (schemaVersion <= 0)
                throw new InvalidOperationException("Custom layer schemaVersion must be positive.");
            if (minimumInputs < 0 || maximumInputs < -1 || (maximumInputs >= 0 && maximumInputs < minimumInputs))
                throw new InvalidOperationException("Custom layer input arity schema is invalid: " + layerType);
            if (minimumOutputs < 0 || maximumOutputs < -1 || (maximumOutputs >= 0 && maximumOutputs < minimumOutputs))
                throw new InvalidOperationException("Custom layer output arity schema is invalid: " + layerType);

            var inputs = layer?.bottomNames?.Length ?? 0;
            var outputs = layer?.topNames?.Length ?? 0;
            if (inputs < minimumInputs || (maximumInputs >= 0 && inputs > maximumInputs))
                throw new InvalidOperationException("Custom layer input arity is invalid | type=" + layerType + " | layer=" + (layer?.name ?? string.Empty));
            if (outputs < minimumOutputs || (maximumOutputs >= 0 && outputs > maximumOutputs))
                throw new InvalidOperationException("Custom layer output arity is invalid | type=" + layerType + " | layer=" + (layer?.name ?? string.Empty));

            foreach (var parameter in parameters ?? Array.Empty<AexisLayerParameterSchema>())
            {
                if (parameter == null)
                    throw new InvalidOperationException("Custom layer schema cannot contain a null parameter.");
                parameter.Validate(layer, layerType);
            }
        }
    }

    [Serializable]
    public sealed class AexisModelExtensionDeclaration
    {
        public string typeName = string.Empty;
        public int schemaVersion = 1;
        public string kernelId = string.Empty;
        public bool textureNativeRequired = true;
    }

    public sealed class AexisCustomLayerDefinition
    {
        public string typeName = string.Empty;
        public string kernelId = string.Empty;
        public AexisCustomLayerSchema schema = new AexisCustomLayerSchema();
        public Func<AexisBaseLayer> createLayer;
    }

    public interface IAexisShaderKernelExtension
    {
        string KernelId { get; }
        void ExecuteRenderTexture(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context);
        void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context);
    }

    // A shader extension is deliberately an execution contract rather than a raw
    // ComputeShader reference. The extension receives the existing Pack4 session and
    // must publish its outputs through texture-native APIs; it never receives a normal
    // activation ComputeBuffer.
    public static class AexisShaderKernelRegistry
    {
        private static readonly object Gate = new object();
        private static readonly Dictionary<string, IAexisShaderKernelExtension> Kernels =
            new Dictionary<string, IAexisShaderKernelExtension>(StringComparer.Ordinal);

        public static void Register(IAexisShaderKernelExtension extension, bool replaceExisting = false)
        {
            if (extension == null)
                throw new ArgumentNullException(nameof(extension));
            if (string.IsNullOrWhiteSpace(extension.KernelId))
                throw new ArgumentException("Shader kernel extensions require KernelId.", nameof(extension));
            lock (Gate)
            {
                if (!replaceExisting && Kernels.ContainsKey(extension.KernelId))
                    throw new InvalidOperationException("A shader kernel extension is already registered: " + extension.KernelId);
                Kernels[extension.KernelId] = extension;
            }
        }

        public static bool Unregister(string kernelId)
        {
            if (string.IsNullOrWhiteSpace(kernelId))
                return false;
            lock (Gate)
                return Kernels.Remove(kernelId);
        }

        public static bool IsRegistered(string kernelId)
        {
            if (string.IsNullOrWhiteSpace(kernelId))
                return false;
            lock (Gate)
                return Kernels.ContainsKey(kernelId);
        }

        public static bool TryGet(string kernelId, out IAexisShaderKernelExtension extension)
        {
            extension = null;
            if (string.IsNullOrWhiteSpace(kernelId))
                return false;
            lock (Gate)
                return Kernels.TryGetValue(kernelId, out extension);
        }
    }

    // Global registration is deliberate: imported NCNN graphs identify layers by type
    // name, while host packages own the actual shader/layer implementation. Validation
    // happens before the factory instantiates a layer so a malformed model cannot reach
    // a dispatch path.
    public static class AexisCustomLayerRegistry
    {
        private static readonly object Gate = new object();
        private static readonly Dictionary<string, AexisCustomLayerDefinition> Definitions =
            new Dictionary<string, AexisCustomLayerDefinition>(StringComparer.Ordinal);

        public static void Register(AexisCustomLayerDefinition definition, bool replaceExisting = false)
        {
            if (definition == null)
                throw new ArgumentNullException(nameof(definition));
            if (string.IsNullOrWhiteSpace(definition.typeName))
                throw new ArgumentException("Custom layer typeName is required.", nameof(definition));
            if (definition.createLayer == null)
                throw new ArgumentException("Custom layer createLayer is required.", nameof(definition));
            if (definition.schema == null)
                throw new ArgumentException("Custom layer schema is required.", nameof(definition));
            if (definition.schema.textureNativeRequired && string.IsNullOrWhiteSpace(definition.kernelId))
                throw new ArgumentException("Texture-native custom layers require kernelId.", nameof(definition));

            lock (Gate)
            {
                if (!replaceExisting && Definitions.ContainsKey(definition.typeName))
                    throw new InvalidOperationException("A custom layer is already registered: " + definition.typeName);
                Definitions[definition.typeName] = definition;
            }
        }

        public static bool Unregister(string typeName)
        {
            if (string.IsNullOrWhiteSpace(typeName))
                return false;
            lock (Gate)
                return Definitions.Remove(typeName);
        }

        public static bool IsRegistered(string typeName)
        {
            if (string.IsNullOrWhiteSpace(typeName))
                return false;
            lock (Gate)
                return Definitions.ContainsKey(typeName);
        }

        public static string[] GetRegisteredTypeNames()
        {
            lock (Gate)
            {
                var names = new string[Definitions.Count];
                Definitions.Keys.CopyTo(names, 0);
                Array.Sort(names, StringComparer.Ordinal);
                return names;
            }
        }

        public static bool TryCreate(AexisGraphModel.Layer layer, out AexisBaseLayer instance)
        {
            instance = null;
            if (layer == null || string.IsNullOrWhiteSpace(layer.typeName))
                return false;

            AexisCustomLayerDefinition definition;
            lock (Gate)
            {
                if (!Definitions.TryGetValue(layer.typeName, out definition))
                    return false;
            }

            definition.schema.Validate(layer, definition.typeName);
            instance = definition.createLayer();
            if (instance == null)
                throw new InvalidOperationException("Custom layer factory returned null: " + definition.typeName);
            if (definition.schema.textureNativeRequired && !instance.SupportsCommandBufferPath)
            {
                throw new InvalidOperationException(
                    "Texture-native custom layer must expose a CommandBuffer path: " + definition.typeName);
            }
            return true;
        }

        public static void ValidateDeclarations(IEnumerable<AexisModelExtensionDeclaration> declarations)
        {
            if (declarations == null)
                return;

            var declared = new HashSet<string>(StringComparer.Ordinal);
            foreach (var declaration in declarations)
            {
                if (declaration == null || string.IsNullOrWhiteSpace(declaration.typeName))
                    throw new InvalidOperationException("Model custom layer declarations require typeName.");
                if (!declared.Add(declaration.typeName))
                    throw new InvalidOperationException("Model custom layer declarations contain duplicate type " + declaration.typeName + ".");
                if (declaration.schemaVersion <= 0)
                    throw new InvalidOperationException("Model custom layer declaration schemaVersion must be positive: " + declaration.typeName);
                if (!AexisLayerFactory.IsRegistered(declaration.typeName))
                    throw new InvalidOperationException("Model declares an unregistered custom layer: " + declaration.typeName);
                if (declaration.textureNativeRequired && string.IsNullOrWhiteSpace(declaration.kernelId))
                    throw new InvalidOperationException("Texture-native model custom layer declarations require kernelId: " + declaration.typeName);
                if (declaration.textureNativeRequired && !AexisShaderKernelRegistry.IsRegistered(declaration.kernelId))
                    throw new InvalidOperationException("Model declares an unregistered texture-native shader kernel: " + declaration.kernelId);
            }
        }
    }
}
