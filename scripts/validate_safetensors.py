# scripts/check_lora_modules.py
import gcsfs
from safetensors import safe_open
import tempfile
import os
from datetime import datetime
import io

# Setup GCS filesystem
fs = gcsfs.GCSFileSystem(token='google_default')

# Create log file path on GCS
log_filename = f"gs://where_you_lora_matters_thesis/logs/lora_module_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_buffer = io.StringIO()  # Buffer to accumulate logs

def log_print(message):
    """Print to console and write to log buffer"""
    print(message)
    log_buffer.write(message + '\n')

def download_safetensors_from_gcs(gcs_path):
    """Download safetensors file from GCS to temp file"""
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors')
    temp_path = temp_file.name
    temp_file.close()
    
    # Download from GCS
    with fs.open(gcs_path, 'rb') as f_in:
        with open(temp_path, 'wb') as f_out:
            f_out.write(f_in.read())
    
    return temp_path

def analyze_lora_adapter(gcs_path, strategy_name):
    """Analyze which modules are in a LoRA adapter"""
    
    log_print(f"\n{'='*80}")
    log_print(f"Analyzing: {strategy_name}")
    log_print(f"GCS Path: {gcs_path}")
    log_print(f"{'='*80}\n")
    
    # Download from GCS
    log_print("Downloading safetensors file from GCS...")
    try:
        local_path = download_safetensors_from_gcs(gcs_path)
        log_print(f"✓ Downloaded to {local_path}")
    except Exception as e:
        log_print(f"❌ Failed to download: {e}")
        return None
    
    try:
        # Load and inspect the adapter
        with safe_open(local_path, framework="pt") as f:
            keys = list(f.keys())
            
            log_print(f"\n📄 Total keys in file: {len(keys)}")
            log_print(f"   Sample keys:")
            for k in keys[:5]:
                log_print(f"      {k}")
            
            # Organize by module type
            modules = {}
            for key in keys:
                # Extract module path (everything before .lora_A or .lora_B)
                if '.lora_A' in key or '.lora_B' in key:
                    # Remove the lora_A/lora_B suffix and any .weight/.default suffixes
                    module_path = key.replace('.lora_A.weight', '').replace('.lora_B.weight', '')
                    module_path = module_path.replace('.lora_A.default.weight', '').replace('.lora_B.default.weight', '')
                    module_path = module_path.replace('.weight', '')
                    
                    if module_path not in modules:
                        modules[module_path] = {'lora_A': False, 'lora_B': False, 'keys': []}
                    
                    modules[module_path]['keys'].append(key)
                    if '.lora_A' in key:
                        modules[module_path]['lora_A'] = True
                    elif '.lora_B' in key:
                        modules[module_path]['lora_B'] = True
            
            # Categorize modules based on YOUR specific target modules
            vision_modules = []
            projector_modules = []
            llm_modules = []
            other_modules = []
            
            for module_path in sorted(modules.keys()):
                # Check for vision encoder modules (visual.blocks.X.attn.qkv or .proj)
                if 'visual.blocks' in module_path and 'attn' in module_path:
                    vision_modules.append(module_path)
                # Check for projector modules (visual.merger or visual.deepstack_merger_list)
                elif 'visual.merger' in module_path or 'visual.deepstack_merger_list' in module_path:
                    projector_modules.append(module_path)
                # Check for LLM modules (q_proj, k_proj, v_proj, o_proj in model.layers)
                elif 'model.layers' in module_path and any(proj in module_path for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                    llm_modules.append(module_path)
                else:
                    other_modules.append(module_path)
            
            # Print summary
            log_print(f"\n📊 Total LoRA modules: {len(modules)}")
            log_print(f"   Vision modules (visual.blocks.X.attn.*): {len(vision_modules)}")
            log_print(f"   Projector modules (visual.merger/deepstack): {len(projector_modules)}")
            log_print(f"   LLM modules (model.layers.X.*_proj): {len(llm_modules)}")
            if other_modules:
                log_print(f"   Other modules: {len(other_modules)}")
            
            # Print detailed breakdown
            if vision_modules:
                log_print(f"\n🖼️  Vision Modules ({len(vision_modules)}):")
                log_print(f"   Expected: 54 modules (27 blocks × 2 types: qkv + proj)")
                for mod in vision_modules[:5]:  # Show first 5
                    log_print(f"   ✓ {mod}")
                if len(vision_modules) > 5:
                    log_print(f"   ... and {len(vision_modules) - 5} more")
            
            if projector_modules:
                log_print(f"\n🔗 Projector Modules ({len(projector_modules)}):")
                log_print(f"   Expected: 8 modules (merger + 3 deepstack mergers × 2 layers each)")
                for mod in projector_modules:
                    log_print(f"   ✓ {mod}")
            
            if llm_modules:
                log_print(f"\n💬 LLM Modules ({len(llm_modules)}):")
                log_print(f"   Expected: 4 × num_layers (q_proj, k_proj, v_proj, o_proj per layer)")
                for mod in llm_modules[:5]:  # Show first 5
                    log_print(f"   ✓ {mod}")
                if len(llm_modules) > 5:
                    log_print(f"   ... and {len(llm_modules) - 5} more")
            
            if other_modules:
                log_print(f"\n❓ Other Modules ({len(other_modules)}):")
                for mod in other_modules[:10]:
                    log_print(f"   ✓ {mod}")
                if len(other_modules) > 10:
                    log_print(f"   ... and {len(other_modules) - 10} more")
            
            # Check for completeness (both lora_A and lora_B)
            incomplete = [m for m, v in modules.items() if not (v['lora_A'] and v['lora_B'])]
            if incomplete:
                log_print(f"\n⚠️  Warning: {len(incomplete)} modules missing lora_A or lora_B:")
                for mod in incomplete[:5]:
                    log_print(f"   - {mod}")
            
            return {
                'total': len(modules),
                'vision': len(vision_modules),
                'projector': len(projector_modules),
                'llm': len(llm_modules),
                'other': len(other_modules)
            }
    
    finally:
        # Clean up temp file
        if os.path.exists(local_path):
            os.remove(local_path)

# --------------------------------
# Main script
# --------------------------------

log_print("="*80)
log_print(f"LoRA Module Verification - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print("="*80)

# Check all four strategies
strategies = {
    "llm_only": "gs://where_you_lora_matters_thesis/artifacts/Qwen3-VL/llm/validation_outputs_final/r32_lr5e-05_run1/checkpoint-6000/adapter_model.safetensors",
    "projector_only": "gs://where_you_lora_matters_thesis/artifacts/Qwen3-VL/projector_only/checkpoint-6000/adapter_model.safetensors",
    "llm_projector": "gs://where_you_lora_matters_thesis/artifacts/Qwen3-VL/llm_proj/validation_outputs_final/llm_proj_r32_64_lr5e-05_0.0001/checkpoint-6000/adapter_model.safetensors",
    "vision_projector": "gs://where_you_lora_matters_thesis/artifacts/Qwen3-VL/vision_proj_experiments/final_experiment_64_1e-4/r64_lr1e-04_run1/checkpoint-6000/adapter_model.safetensors"
}

log_print("\n🔍 Checking LoRA adapters for all strategies...\n")
log_print("Expected configurations based on your target modules:")
log_print("  - llm_only: Should have 144 LLM modules (4 proj types × 36 layers)")
log_print("  - projector_only: Should have 8 Projector modules")
log_print("  - llm_projector: Should have 144 LLM + 8 Projector = 152 modules")
log_print("  - vision_projector: Should have 54 Vision + 8 Projector = 62 modules")

results = {}

for strategy_name, gcs_path in strategies.items():
    result = analyze_lora_adapter(gcs_path, strategy_name)
    if result:
        results[strategy_name] = result

# --------------------------------
# Final summary
# --------------------------------

if results:
    log_print("\n" + "="*80)
    log_print("📋 SUMMARY TABLE")
    log_print("="*80)
    log_print(f"{'Strategy':<20} {'Vision':<10} {'Projector':<12} {'LLM':<10} {'Other':<10} {'Total':<10}")
    log_print("-"*80)
    
    for strategy_name in ['llm_only', 'projector_only', 'llm_projector', 'vision_projector']:
        if strategy_name in results:
            r = results[strategy_name]
            log_print(f"{strategy_name:<20} {r['vision']:<10} {r['projector']:<12} {r['llm']:<10} {r['other']:<10} {r['total']:<10}")
    
    log_print("\n" + "="*80)
    log_print("✅ VERIFICATION COMPLETE!")
    log_print("="*80)
    
    # Validate expected configurations
    log_print("\n🔎 Configuration Validation:")
    
    errors = []
    
    if 'llm_only' in results:
        r = results['llm_only']
        if r['llm'] > 0 and r['vision'] == 0 and r['projector'] == 0:
            log_print(f"  ✅ llm_only: Correct (only LLM modules: {r['llm']})")
        else:
            log_print(f"  ❌ llm_only: INCORRECT (vision={r['vision']}, projector={r['projector']}, llm={r['llm']})")
            errors.append('llm_only')
    
    if 'projector_only' in results:
        r = results['projector_only']
        if r['projector'] > 0 and r['vision'] == 0 and r['llm'] == 0:
            log_print(f"  ✅ projector_only: Correct (only Projector modules: {r['projector']})")
        else:
            log_print(f"  ❌ projector_only: INCORRECT (vision={r['vision']}, projector={r['projector']}, llm={r['llm']})")
            errors.append('projector_only')
    
    if 'vision_projector' in results:
        r = results['vision_projector']
        if r['vision'] > 0 and r['projector'] > 0 and r['llm'] == 0:
            log_print(f"  ✅ vision_projector: Correct (Vision={r['vision']} + Projector={r['projector']})")
        else:
            log_print(f"  ❌ vision_projector: INCORRECT (vision={r['vision']}, projector={r['projector']}, llm={r['llm']})")
            errors.append('vision_projector')
    
    if 'llm_projector' in results:
        r = results['llm_projector']
        if r['llm'] > 0 and r['projector'] > 0 and r['vision'] == 0:
            log_print(f"  ✅ llm_projector: Correct (LLM={r['llm']} + Projector={r['projector']})")
        else:
            log_print(f"  ❌ llm_projector: INCORRECT (vision={r['vision']}, projector={r['projector']}, llm={r['llm']})")
            errors.append('llm_projector')
    
    if errors:
        log_print(f"\n⚠️  WARNING: {len(errors)} configuration(s) may be incorrect!")
        log_print("   Please check your training config and re-train if needed.")
    else:
        log_print("\n✅ All configurations are correct!")
    
    # Add thesis insight about projector-only vs vision+projector
    if 'projector_only' in results and 'vision_projector' in results:
        log_print("\n" + "="*80)
        log_print("💡 THESIS INSIGHT")
        log_print("="*80)
        log_print("If projector_only and vision_projector achieve similar VQA accuracy,")
        log_print("this suggests that:")
        log_print("  1. The projector is the critical bottleneck for VQA task")
        log_print("  2. The pretrained vision encoder is already well-aligned")
        log_print("  3. Fine-tuning vision encoder doesn't provide additional benefit")
        log_print("\nThis is a valuable finding for understanding modality fusion!")

else:
    log_print("\n❌ No adapters found to analyze")

# Save log to GCS
print(f"\n💾 Saving log to GCS...")
with fs.open(log_filename, 'w') as f:
    f.write(log_buffer.getvalue())

print(f"📄 Log file saved to: {log_filename}")
print(f"   View with: gsutil cat {log_filename}")