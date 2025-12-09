import os
import random
import re
import yaml
import csv
import fnmatch
import folder_paths
import server
from aiohttp import web

# ==============================================================================
# GLOBAL CACHE
# ==============================================================================
GLOBAL_CACHE = {}
# Persist the card index across generations
GLOBAL_INDEX = {'built': False, 'files': set(), 'entries': {}}
ALL_KEY = 'all_files_index'

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_tag(tag):
    if tag is None:
        return ""
    tag = tag.replace("__", "").replace('<', '').replace('>', '').strip()
    if tag.startswith('#'):
        return tag
    return tag

def read_file_lines(file):
    f_lines = file.read().splitlines()
    lines = []
    for line in f_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        if '#' in line:
            line = line.split('#')[0].strip()
        lines.append(line)
    return lines

def parse_wildcard_range(range_str, num_variants):
    if range_str is None:
        return 1, 1
    
    if "-" in range_str:
        parts = range_str.split("-")
        if len(parts) == 2:
            start = int(parts[0]) if parts[0] else 1
            end = int(parts[1]) if parts[1] else num_variants
            return min(start, end), max(start, end)
    
    try:
        val = int(range_str)
        return val, val
    except:
        return 1, 1

def process_wildcard_range(tag, lines):
    if not lines:
        return ""
    if tag.startswith('#'):
        return None
    
    if "$$" not in tag:
        selected = random.choice(lines)
        if '#' in selected:
            selected = selected.split('#')[0].strip()
        return selected
        
    range_str, tag_name = tag.split("$$", 1)
    try:
        low, high = parse_wildcard_range(range_str, len(lines))
        num_items = random.randint(low, high)
        if num_items == 0:
            return ""
            
        selected = random.sample(lines, min(num_items, len(lines)))
        selected = [line.split('#')[0].strip() if '#' in line else line for line in selected]
        return ", ".join(selected)
    except Exception as e:
        print(f"Error processing wildcard range: {e}")
        selected = random.choice(lines)
        if '#' in selected:
            selected = selected.split('#')[0].strip()
        return selected

def get_all_wildcard_paths():
    """
    Returns a list of all valid wildcard directory paths.
    """
    paths = set()
    
    # 1. Internal Umi Folder
    internal_path = os.path.join(os.path.dirname(__file__), "wildcards")
    if os.path.exists(internal_path):
        paths.add(internal_path)
    
    # 2. ComfyUI/wildcards (Root)
    root_wildcards = os.path.join(folder_paths.base_path, "wildcards")
    if os.path.exists(root_wildcards):
        paths.add(root_wildcards)

    # 3. ComfyUI/models/wildcards
    models_wildcards = os.path.join(folder_paths.models_dir, "wildcards")
    if os.path.exists(models_wildcards):
        paths.add(models_wildcards)

    # 4. Check for 'wildcards' type in folder_paths (external extensions)
    try:
        ext_paths = folder_paths.get_folder_paths("wildcards")
        if ext_paths:
            for p in ext_paths:
                if os.path.exists(p):
                    paths.add(p)
    except:
        pass
    
    return list(paths)

# ==============================================================================
# LOGIC EVALUATOR (Safe replacement for eval)
# ==============================================================================
class SafeLogicEvaluator:
    def __init__(self):
        self.precedence = {'OR': 1, 'XOR': 1, 'AND': 2, 'NOT': 3}

    def tokenize(self, expression):
        # Add spaces around parens for easier splitting
        expression = expression.replace('(', ' ( ').replace(')', ' ) ')
        return expression.split()

    def to_rpn(self, tokens):
        output_queue = []
        operator_stack = []
        
        for token in tokens:
            token_upper = token.upper()
            
            if token_upper in ('TRUE', 'FALSE'):
                output_queue.append(token_upper == 'TRUE')
            elif token_upper == 'NOT':
                # NOT is right-associative: only pop operators with strictly higher precedence (>)
                # This ensures consecutive NOTs don't pop each other (equal precedence stays)
                while (operator_stack and operator_stack[-1] != '(' and
                       self.precedence.get(operator_stack[-1], 0) > self.precedence[token_upper]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token_upper)
            elif token_upper in ('AND', 'OR', 'XOR'):
                # Binary operators are left-associative: pop operators with >= precedence
                while (operator_stack and operator_stack[-1] != '(' and
                       self.precedence.get(operator_stack[-1], 0) >= self.precedence[token_upper]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token_upper)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()
            else:
                # Should not happen if pre-processed correctly, but treat as False safe fallback
                pass
                
        while operator_stack:
            output_queue.append(operator_stack.pop())
            
        return output_queue

    def evaluate_rpn(self, rpn_queue):
        stack = []
        for token in rpn_queue:
            if isinstance(token, bool):
                stack.append(token)
            elif token == 'NOT':
                if stack:
                    stack.append(not stack.pop())
            elif token == 'AND':
                if len(stack) >= 2:
                    val2 = stack.pop()
                    val1 = stack.pop()
                    stack.append(val1 and val2)
            elif token == 'OR':
                if len(stack) >= 2:
                    val2 = stack.pop()
                    val1 = stack.pop()
                    stack.append(val1 or val2)
            elif token == 'XOR':
                if len(stack) >= 2:
                    val2 = stack.pop()
                    val1 = stack.pop()
                    stack.append(val1 != val2)
        
        return stack[0] if stack else False

    def evaluate(self, expression_tokens):
        # Expects a list of strings that are either operators or "True"/"False"
        try:
            rpn = self.to_rpn(expression_tokens)
            return self.evaluate_rpn(rpn)
        except Exception:
            return False

# ==============================================================================
# CORE CLASSES
# ==============================================================================

class TagLoader:
    def __init__(self, wildcard_paths, options):
        # Allow passing a single string or a list
        if isinstance(wildcard_paths, str):
            self.wildcard_locations = [wildcard_paths]
        else:
            self.wildcard_locations = wildcard_paths

        self.loaded_tags = {}
        self.yaml_entries = {}
        self.files_index = set() 
        self.index_built = False
        self.ignore_paths = options.get('ignore_paths', True)
        self.verbose = options.get('verbose', False)
        
        # Mappings
        self.txt_lookup = {}
        self.yaml_lookup = {}
        self.csv_lookup = {}
        
        self.refresh_maps()

    def refresh_maps(self):
        self.txt_lookup = {}
        self.yaml_lookup = {}
        self.csv_lookup = {}
        
        for location in self.wildcard_locations:
            if not os.path.exists(location):
                continue
                
            for root, dirs, files in os.walk(location):
                for file in files:
                    full_path = os.path.join(root, file)
                    # Rel path must be relative to the *current* root location being scanned
                    rel_path = os.path.relpath(full_path, location)
                    
                    # Normalize slashes for consistency across OS
                    key = os.path.splitext(rel_path)[0].replace(os.sep, '/')
                    
                    name_lower = file.lower()
                    if name_lower.endswith('.txt'):
                        self.txt_lookup[key.lower()] = full_path
                    elif name_lower.endswith('.yaml'):
                        self.yaml_lookup[key.lower()] = full_path
                    elif name_lower.endswith('.csv'):
                        self.csv_lookup[key.lower()] = full_path

    def build_index(self):
        # OPTIMIZATION: Check if we already did this work globally
        if GLOBAL_INDEX['built']:
            self.files_index = GLOBAL_INDEX['files']
            self.yaml_entries = GLOBAL_INDEX['entries']
            self.index_built = True
            return

        if self.index_built:
            return

        new_index = set()
        new_entries = {}
        
        # 1. Add Files (TXT/CSV)
        for key in self.txt_lookup.keys():
            new_index.add(key)
        for key in self.csv_lookup.keys():
            new_index.add(key)

        # 2. Add YAML Keys
        for file_key, full_path in self.yaml_lookup.items():
            if file_key == 'globals':
                continue
            try:
                with open(full_path, encoding="utf8") as f:
                    data = yaml.safe_load(f)
                    
                    # Fix: Immediate population of entries for Umi format
                    if self.is_umi_format(data):
                         for k, v in data.items():
                             new_index.add(k)
                             # Store card data
                             if isinstance(v, dict):
                                 processed = self.process_yaml_entry(k, v)
                                 if processed['tags']:
                                     new_entries[k.lower()] = processed
                    else:
                        flat_data = self.flatten_hierarchical_yaml(data)
                        for k in flat_data.keys():
                            combined = f"{file_key}/{k}"
                            new_index.add(combined)
            except Exception as e:
                pass

        # Save to Instance
        self.files_index = new_index
        self.yaml_entries = new_entries
        self.index_built = True
        
        # Save to Global Cache (Persist for next run)
        GLOBAL_INDEX['files'] = new_index
        GLOBAL_INDEX['entries'] = new_entries
        GLOBAL_INDEX['built'] = True

    def load_globals(self):
        """Loads globals.yaml from ALL detected paths and merges them."""
        merged_globals = {}
        
        for location in self.wildcard_locations:
            global_path = os.path.join(location, 'globals.yaml')
            if os.path.exists(global_path):
                try:
                    with open(global_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, dict):
                            # Update dictionary (later paths overwrite earlier ones)
                            merged_globals.update({str(k): str(v) for k, v in data.items()})
                except Exception as e:
                    print(f"[UmiAI] Error loading globals.yaml at {global_path}: {e}")
                    
        return merged_globals

    def process_yaml_entry(self, title, entry_data):
        return {
            'title': title,
            'description': entry_data.get('Description', [None])[0] if isinstance(entry_data.get('Description', []), list) else None,
            'prompts': entry_data.get('Prompts', []),
            'prefixes': entry_data.get('Prefix', []),
            'suffixes': entry_data.get('Suffix', []),
            'tags': [x.lower().strip() for x in entry_data.get('Tags', [])]
        }
    
    def flatten_hierarchical_yaml(self, data, prefix=""):
        results = {}
        if isinstance(data, dict):
            for k, v in data.items():
                clean_key = str(k).strip()
                new_prefix = f"{prefix}/{clean_key}" if prefix else clean_key
                results.update(self.flatten_hierarchical_yaml(v, new_prefix))
        elif isinstance(data, list):
            clean_list = [str(x) for x in data if x is not None]
            results[prefix] = clean_list
        elif data is not None:
            results[prefix] = [str(data)]
        return results

    def is_umi_format(self, data):
        """
        Determines if the YAML file follows the Umi 'Card' structure.
        Criteria: The data is a dictionary, and at least one of its values 
        is a dictionary containing a 'Prompts' key (case-insensitive).
        """
        if not isinstance(data, dict):
            return False
        
        # We iterate through the top-level keys (Card Titles)
        for key, value in data.items():
            if isinstance(value, dict):
                # Check keys case-insensitively for 'prompts'
                keys_lower = {k.lower() for k in value.keys()}
                if 'prompts' in keys_lower:
                    return True
        return False

    def load_tags(self, requested_tag, verbose=False):
        # Fix: Handle Global Index Request for Tag Aggregation
        if requested_tag == ALL_KEY:
            self.build_index() # Ensure we have data
            return self.yaml_entries

        if requested_tag in GLOBAL_CACHE:
            return GLOBAL_CACHE[requested_tag]
        
        lower_tag = requested_tag.lower()
        
        # 1. Try TXT files
        if lower_tag in self.txt_lookup:
            with open(self.txt_lookup[lower_tag], encoding="utf8") as f:
                lines = read_file_lines(f)
                GLOBAL_CACHE[requested_tag] = lines
                return lines
        
        # 2. Try CSV files
        if lower_tag in self.csv_lookup:
            with open(self.csv_lookup[lower_tag], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                GLOBAL_CACHE[requested_tag] = rows
                return rows

        # 3. YAML Handling (Branching Path)
        parts = lower_tag.split('/')
        found_file = None
        key_suffix = ""

        if lower_tag in self.yaml_lookup:
            found_file = self.yaml_lookup[lower_tag]
        else:
            for i in range(len(parts) - 1, 0, -1):
                potential_file = "/".join(parts[:i])
                potential_key = "/".join(parts[i:])
                if potential_file in self.yaml_lookup:
                    found_file = self.yaml_lookup[potential_file]
                    key_suffix = potential_key
                    break
        
        if found_file:
            with open(found_file, encoding="utf8") as file:
                try:
                    data = yaml.safe_load(file)
                    
                    # SYSTEM 1: Umi "Card" System
                    if self.is_umi_format(data):
                        # Populate global index if not already (mostly handled by build_index now)
                        for title, entry in data.items():
                            if isinstance(entry, dict):
                                processed = self.process_yaml_entry(title, entry)
                                if processed['tags']:
                                    self.yaml_entries[title.lower()] = processed

                        # Return specific card if requested
                        if key_suffix:
                             for k, v in data.items():
                                 if k.lower() == key_suffix:
                                     processed = self.process_yaml_entry(k, v)
                                     GLOBAL_CACHE[requested_tag] = processed['prompts']
                                     return processed['prompts']
                        return []

                    # SYSTEM 2: Standard "Folder" System (Hierarchical)
                    else:
                        flat_data = self.flatten_hierarchical_yaml(data)
                        if key_suffix:
                            # Exact path match (case-insensitive)
                            for k, v in flat_data.items():
                                if k.lower() == key_suffix:
                                    GLOBAL_CACHE[requested_tag] = v
                                    return v
                            return []
                        else:
                            # Return all leaves if root file selected
                            all_values = []
                            for v in flat_data.values():
                                all_values.extend(v)
                            return all_values

                except Exception as e:
                    if verbose: print(f'Error parsing YAML {found_file}: {e}')

        return []

    def get_glob_matches(self, pattern):
        self.build_index()
        return fnmatch.filter(self.files_index, pattern)

    def get_entry_details(self, title):
        # Fallback if entry is not in dict but index is built
        if title and title.lower() in self.yaml_entries:
            return self.yaml_entries[title.lower()]
        return self.yaml_entries.get(title)

class TagSelector:
    def __init__(self, tag_loader, options):
        self.tag_loader = tag_loader
        self.previously_selected_tags = {}
        self.used_values = {}
        self.selected_options = options.get('selected_options', {})
        self.verbose = options.get('verbose', False)
        self.global_seed = options.get('seed', 0)
        self.seeded_values = {}
        self.processing_stack = set()
        self.resolved_seeds = {}
        self.selected_entries = {}
        self.variables = {} 
        self.evaluator = SafeLogicEvaluator()

    def update_variables(self, variables):
        self.variables = variables

    def clear_seeded_values(self):
        self.seeded_values = {}
        self.resolved_seeds = {}
        self.processing_stack.clear()
        self.selected_entries.clear()

    def evaluate_criteria(self, criteria_str, candidate_tags):
        """
        Evaluates a logic string (e.g. "red OR (blue AND NOT green)") 
        against a set/list of candidate tags.
        """
        # 1. Pre-process Legacy Syntax
        clean_criteria = criteria_str.replace(',', ' AND ')
        clean_criteria = re.sub(r'(^|\s)--', r' NOT ', clean_criteria)
        
        # 2. Tokenize logic
        ops = {'AND': 'and', 'OR': 'or', 'NOT': 'not', 'XOR': '!='}
        tokens = re.split(r'(\(|\)|\bAND\b|\bOR\b|\bNOT\b|\bXOR\b)', clean_criteria, flags=re.IGNORECASE)
        
        expression_tokens = []
        candidate_context = " ".join(candidate_tags).lower()

        for token in tokens:
            token = token.strip()
            if not token: continue
            
            upper_token = token.upper()
            if upper_token in ops:
                expression_tokens.append(upper_token) # Normalize to uppercase for evaluator
            elif token in ('(', ')'):
                expression_tokens.append(token)
            else:
                # FIX: Explicit Variable Equality Check (e.g. $type=fire)
                if '=' in token:
                    left, right = token.split('=', 1)
                    left = left.strip()
                    right = right.strip()
                    
                    if left.startswith('$') and left[1:] in self.variables:
                        left_val = str(self.variables[left[1:]]).lower()
                        result = left_val == right.lower()
                    else:
                        result = left.lower() == right.lower()
                    expression_tokens.append(str(result))
                         
                # Standard Variable Resolution (e.g. $type)
                elif token.startswith('$') and token[1:] in self.variables:
                    token_val = str(self.variables[token[1:]])
                    exists = token_val.lower() in candidate_context
                    expression_tokens.append(str(exists))
                
                # Standard Text Search
                else:
                    clean_token = token.replace('[','').replace(']','').strip()
                    exists = clean_token.lower() in candidate_context
                    expression_tokens.append(str(exists))

        return self.evaluator.evaluate(expression_tokens)

    def get_tag_choice(self, parsed_tag, tags):
        if isinstance(tags, list) and len(tags) > 0 and isinstance(tags[0], dict):
            row = random.choice(tags)
            vars_out = []
            for k, v in row.items():
                vars_out.append(f"${k.strip()}={v.strip()}")
            return " ".join(vars_out)

        if not isinstance(tags, list):
            return ""
        
        seed_match = re.match(r'#([0-9|]+)\$\$(.*)', parsed_tag)
        if seed_match:
            seed_options = seed_match.group(1).split('|')
            chosen_seed = random.choice(seed_options)
            
            if chosen_seed in self.seeded_values:
                selected = self.seeded_values[chosen_seed]
                return self.resolve_wildcard_recursively(selected, chosen_seed)
            
            unused = [t for t in tags if t not in self.used_values]
            selected = random.choice(unused) if unused else random.choice(tags)
            
            self.seeded_values[chosen_seed] = selected
            self.used_values[selected] = True
            return self.resolve_wildcard_recursively(selected, chosen_seed)

        selected = None
        if len(tags) == 1:
            selected = tags[0]
        else:
            unused = [t for t in tags if t not in self.used_values]
            selected = random.choice(unused) if unused else random.choice(tags)

        if selected:
            self.used_values[selected] = True
            entry_details = self.tag_loader.get_entry_details(selected)
            if entry_details:
                self.selected_entries[parsed_tag] = entry_details
                if entry_details['prompts']:
                    selected = random.choice(entry_details['prompts'])
            if isinstance(selected, str) and '#' in selected:
                selected = selected.split('#')[0].strip()

        return selected

    def resolve_wildcard_recursively(self, value, seed_id=None):
        if value.startswith('__') and value.endswith('__'):
            nested_tag = value[2:-2]
            nested_seed = f"{seed_id}_{nested_tag}" if seed_id else None
            
            if nested_tag in self.processing_stack:
                return value
            self.processing_stack.add(nested_tag)
            
            if nested_seed and nested_seed in self.resolved_seeds:
                resolved = self.resolved_seeds[nested_seed]
            else:
                resolved = self.select(nested_tag)
                if nested_seed:
                    self.resolved_seeds[nested_seed] = resolved
            
            self.processing_stack.remove(nested_tag)
            return resolved
        return value

    def get_tag_group_choice(self, parsed_tag, criteria_str, tags):
        if not isinstance(tags, dict):
            return ""
        
        candidates = []
        for title, entry_data in tags.items():
            candidate_tags = []
            if isinstance(entry_data, dict):
                candidate_tags = entry_data.get('tags', [])
            elif isinstance(entry_data, (list, set)):
                candidate_tags = list(entry_data)
            
            if self.evaluate_criteria(criteria_str, candidate_tags):
                candidates.append(title)

        if candidates:
            seed_match = re.match(r'#([0-9|]+)\$\$(.*)', parsed_tag)
            seed_id = seed_match.group(1) if seed_match else None
            
            selected_title = random.choice(candidates)
            if seed_id and seed_id in self.seeded_values:
                selected_title = self.seeded_values[seed_id]
            elif seed_id:
                self.seeded_values[seed_id] = selected_title
                
            entry_details = self.tag_loader.get_entry_details(selected_title)
            if entry_details:
                self.selected_entries[parsed_tag] = entry_details
                if entry_details['prompts']:
                    return self.resolve_wildcard_recursively(random.choice(entry_details['prompts']), seed_id)
            return self.resolve_wildcard_recursively(selected_title, seed_id)
        return ""

    def select(self, tag, groups=None):
        self.previously_selected_tags.setdefault(tag, 0)
        if self.previously_selected_tags.get(tag) > 500:
            return f"LOOP_ERROR({tag})"
        
        self.previously_selected_tags[tag] += 1
        parsed_tag = parse_tag(tag)
        
        # --- GLOB MATCHING ---
        if '*' in parsed_tag or '?' in parsed_tag:
            matches = self.tag_loader.get_glob_matches(parsed_tag)
            if matches:
                random.shuffle(matches)
                for selected_key in matches:
                    result = self.select(selected_key, groups)
                    if result and str(result).strip():
                        return result
            return ""

        sequential = False
        if parsed_tag.startswith('~'):
            sequential = True
            parsed_tag = parsed_tag[1:]

        if '$$' in parsed_tag and not parsed_tag.startswith('#'):
            range_part, file_part = parsed_tag.split('$$', 1)
            if any(c.isdigit() for c in range_part) or '-' in range_part:
                tags = self.tag_loader.load_tags(file_part, self.verbose)
                if isinstance(tags, list):
                    return process_wildcard_range(parsed_tag, tags)

        if parsed_tag.startswith('#'):
            tags = self.tag_loader.load_tags(parsed_tag.split('$$')[1], self.verbose)
            if isinstance(tags, list):
                return self.get_tag_choice(parsed_tag, tags)

        tags = self.tag_loader.load_tags(parsed_tag, self.verbose)
        
        if sequential and isinstance(tags, list) and tags:
            idx = self.global_seed % len(tags)
            selected = tags[idx]
            if isinstance(selected, dict):
                 vars_out = []
                 for k, v in selected.items():
                     vars_out.append(f"${k.strip()}={v.strip()}")
                 return " ".join(vars_out)
            if '#' in selected:
                selected = selected.split('#')[0].strip()
            return self.resolve_wildcard_recursively(selected, self.global_seed)

        if groups:
            return self.get_tag_group_choice(parsed_tag, groups, tags)
        if tags:
            return self.get_tag_choice(parsed_tag, tags)
        
        return None 

    def get_prefixes_and_suffixes(self):
        prefixes, suffixes = [], []
        for entry in self.selected_entries.values():
            for p in entry.get('prefixes', []):
                if not p:
                    continue
                p_str = str(p)
                prefixes.append(p_str)
            for s in entry.get('suffixes', []):
                if not s:
                    continue
                s_str = str(s)
                suffixes.append(s_str)
        return {'prefixes': prefixes, 'suffixes': suffixes}

class TagReplacer:
    def __init__(self, tag_selector):
        self.tag_selector = tag_selector
        self.wildcard_regex = re.compile(r'(__|<)(.*?)(__|>)')
        self.clean_regex = re.compile(r'\[clean:(.*?)\]', re.IGNORECASE)
        self.shuffle_regex = re.compile(r'\[shuffle:(.*?)\]', re.IGNORECASE)

    def replace_wildcard(self, matches):
        if not matches or len(matches.groups()) != 3:
            return ""
        match = matches.group(2)
        if not match:
            return ""
        
        if ':' in match:
            scope, opts = match.split(':', 1)
            if opts.strip():
                 selected = self.tag_selector.select(scope, opts)
            else:
                 selected = self.tag_selector.select(scope)
        else:
            opts_regexp = re.findall(r'(?<=\[)(.*?)(?=\])', match)
            if opts_regexp:
                selected = self.tag_selector.select(ALL_KEY, ",".join(opts_regexp))
            else:
                selected = self.tag_selector.select(match)
        
        if selected is not None:
            if isinstance(selected, str) and '#' in selected:
                selected = selected.split('#')[0].strip()
            return str(selected) 
            
        return matches.group(0)

    def replace_functions(self, prompt):
        def _shuffle(match):
            content = match.group(1)
            items = [x.strip() for x in content.split(',')]
            random.shuffle(items)
            return ", ".join(items)
        
        def _clean(match):
            content = match.group(1)
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r',\s*,', ',', content)
            content = content.replace(' ,', ',')
            return content.strip(', ')

        p = self.shuffle_regex.sub(_shuffle, prompt)
        p = self.clean_regex.sub(_clean, p)
        return p

    def replace(self, prompt):
        p = self.wildcard_regex.sub(self.replace_wildcard, prompt)
        count = 0
        while p != prompt and count < 10:
            prompt = p
            p = self.wildcard_regex.sub(self.replace_wildcard, prompt)
            count += 1
        p = self.replace_functions(p)
        return p

class DynamicPromptReplacer:
    def __init__(self, seed):
        self.seed = seed

    def find_innermost_braces(self, text):
        """
        Find the position of the innermost {...} block.
        Returns (start, end, content) or None if no braces found.
        """
        depth = 0
        start = -1
        innermost_start = -1
        innermost_end = -1
        max_depth = 0
        
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
                if depth > max_depth:
                    max_depth = depth
                    innermost_start = i
            elif char == '}':
                if depth > 0:
                    depth -= 1
                    if depth == max_depth - 1:
                        innermost_end = i
                        # Found an innermost block
                        content = text[innermost_start + 1:innermost_end]
                        return (innermost_start, innermost_end + 1, content)
        
        # If no nested braces, find first simple block
        depth = 0
        for i, char in enumerate(text):
            if char == '{':
                start = i
                depth += 1
            elif char == '}' and depth > 0:
                depth -= 1
                if depth == 0:
                    content = text[start + 1:i]
                    return (start, i + 1, content)
        
        return None

    def weighted_choice(self, content):
        """
        Parse {5::red|3::blue|green} style weighted choices.
        Items without :: get weight=1.
        """
        variants = [s.strip() for s in content.split("|")]
        if not variants:
            return ""
        
        weights = []
        items = []
        
        for variant in variants:
            if '::' in variant:
                parts = variant.split('::', 1)
                try:
                    weight = float(parts[0].strip())
                    item = parts[1].strip()
                    weights.append(weight)
                    items.append(item)
                except (ValueError, IndexError):
                    # Invalid weight format, treat as weight=1
                    weights.append(1.0)
                    items.append(variant)
            else:
                weights.append(1.0)
                items.append(variant)
        
        if not items:
            return ""
        
        # Use random.choices for weighted random selection
        selected = random.choices(items, weights=weights, k=1)
        return selected[0]

    def process_content(self, content):
        """
        Process the content of a {...} block.
        Handles ~, %, $$, ::, and simple choices.
        """
        if not content:
            return ""
        
        # Sequential mode (~)
        if content.startswith('~'):
            content = content[1:]
            if '$$' in content:
                pass  # Fall through to $$ handling
            else:
                variants = [s.strip() for s in content.split("|")]
                if not variants:
                    return ""
                return variants[self.seed % len(variants)]

        # Weighted selection (::)
        if '::' in content and '$$' not in content:
            return self.weighted_choice(content)

        # Probability (%)
        if '%' in content and '$$' not in content and '::' not in content:
            parts = content.split('%', 1)
            try:
                chance = float(parts[0])
                options = parts[1].split('|')
                if random.random() * 100 < chance:
                    return options[0]
                elif len(options) > 1:
                    return random.choice(options[1:])
                else:
                    return ""
            except ValueError:
                pass

        # Multi-select with optional custom separator ($$)
        if '$$' in content:
            parts = content.split('$$')
            range_str = parts[0].strip()
            separator = ', '  # default separator
            variants_str = parts[-1]
            
            # If 3 parts, middle part is custom separator
            if len(parts) == 3:
                separator = parts[1]
            
            variants = [s.strip() for s in variants_str.split("|")]
            low, high = parse_wildcard_range(range_str, len(variants))
            count = random.randint(low, high)
            if count <= 0:
                return ""
            selected = random.sample(variants, min(count, len(variants)))
            return separator.join(selected)

        # Simple random choice
        variants = [s.strip() for s in content.split("|")]
        if not variants:
            return ""
        return random.choice(variants)

    def replace(self, template):
        """
        Recursively process from innermost braces outward.
        This allows arbitrary nesting depth.
        """
        if not template:
            return ""
        
        # Maximum iterations to prevent infinite loops
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations:
            result = self.find_innermost_braces(template)
            if result is None:
                # No more braces to process
                break
            
            start, end, content = result
            replacement = self.process_content(content)
            template = template[:start] + replacement + template[end:]
            iteration += 1
        
        return template

class ConditionalReplacer:
    def __init__(self):
        self.regex = re.compile(
            r'\[if\s+([^:|\]]+?)\s*:\s*((?:(?!\[if).)*?)(?:\s*\|\s*((?:(?!\[if).)*?))?\]', 
            re.IGNORECASE | re.DOTALL
        )
        self.evaluator = SafeLogicEvaluator()

    def evaluate_logic(self, condition, context, variables=None):
        if variables is None: variables = {}
        
        ops = {'AND': 'and', 'OR': 'or', 'NOT': 'not', 'XOR': '!='}
        tokens = re.split(r'(\(|\)|\bAND\b|\bOR\b|\bNOT\b|\bXOR\b)', condition, flags=re.IGNORECASE)
        expression_tokens = []
        
        for token in tokens:
            token = token.strip()
            if not token: continue
            upper_token = token.upper()
            if upper_token in ops:
                expression_tokens.append(upper_token)
            elif token in ('(', ')'):
                expression_tokens.append(token)
            else:
                # FIX: Handle Equality explicitly ($char=robot)
                if '=' in token:
                    left, right = token.split('=', 1)
                    left = left.strip()
                    right = right.strip()
                    
                    if left.startswith('$'):
                        var_name = left[1:]
                        left_val = str(variables.get(var_name, "")).lower()
                    else:
                        left_val = left.lower()
                    
                    expression_tokens.append(str(left_val == right.lower()))
                
                # FIX: Handle Boolean Variable Check ($is_robot)
                elif token.startswith('$'):
                    var_name = token[1:]
                    val = variables.get(var_name, False)
                    # Truthy check (string "false" or "0" becomes False)
                    is_true = bool(val) and str(val).lower() not in ['false', '0', 'no']
                    expression_tokens.append(str(is_true))
                    
                else:
                    exists = token.lower() in context.lower()
                    expression_tokens.append(str(exists))
        
        return self.evaluator.evaluate(expression_tokens)

    def replace(self, prompt, variables):
        while True:
            match = self.regex.search(prompt)
            if not match: break
            
            full_tag = match.group(0)
            condition = match.group(1).strip()
            true_text = match.group(2)
            false_text = match.group(3) if match.group(3) else ""
            
            context = prompt.replace(full_tag, "")

            # FIX: Pass variables into logic engine
            if self.evaluate_logic(condition, context, variables):
                replacement = true_text
            else:
                replacement = false_text
            
            prompt = prompt.replace(full_tag, replacement, 1)
        return prompt

class VariableReplacer:
    def __init__(self):
        # FIX: Updated Regex to be Multiline Anchored and more robust for values
        self.assign_regex = re.compile(r'^\$([a-zA-Z0-9_]+)\s*=\s*(.*?)$', re.MULTILINE)
        self.use_regex = re.compile(r'\$([a-zA-Z0-9_]+)((?:\.[a-zA-Z_]+)*)')
        self.variables = {}

    def load_globals(self, globals_dict):
        self.variables.update(globals_dict)

    def store_variables(self, text, tag_replacer, dynamic_replacer):
        def _replace_assign(match):
            var_name = match.group(1)
            raw_value = match.group(2).strip() # Strip whitespace from captured value
            
            resolved_value = raw_value
            for _ in range(10): 
                prev_value = resolved_value
                resolved_value = tag_replacer.replace(resolved_value)
                resolved_value = dynamic_replacer.replace(resolved_value)
                if prev_value == resolved_value:
                    break
            
            self.variables[var_name] = resolved_value
            return "" 
        return self.assign_regex.sub(_replace_assign, text)

    def replace_variables(self, text):
        def _replace_use(match):
            var_name = match.group(1)
            methods_str = match.group(2) 
            
            value = self.variables.get(var_name)
            if value is None:
                return match.group(0)
            
            if methods_str:
                methods = methods_str.split('.')[1:]
                for method in methods:
                    if method == 'clean':
                        value = value.replace('_', ' ').replace('-', ' ')
                    elif method == 'upper':
                        value = value.upper()
                    elif method == 'lower':
                        value = value.lower()
                    elif method == 'title':
                        value = value.title()
                    elif method == 'capitalize':
                        value = value.capitalize()
                        
            return value
            
        return self.use_regex.sub(_replace_use, text)

# ==============================================================================
# NODE DEFINITION
# ==============================================================================

class ABWildcardNode:
    def __init__(self):
        self.loaded = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "AB Utils"
    COLOR = "#322947"

    # --- SAFETY HELPER ---
    def get_val(self, kwargs, key, default, value_type=None):
        val = kwargs.get(key, default)
        
        # Handle None explicitly (optional inputs can be None)
        if val is None:
            return default
        
        # Type enforcement
        if value_type and not isinstance(val, value_type):
            try:
                if value_type == int:
                    return int(val)
                if value_type == float:
                    return float(val)
                if value_type == str:
                    return str(val)
            except (ValueError, TypeError):
                return default
        
        return val

    def process(self, **kwargs):
        # 1. EXTRACT INPUTS SAFELY (CRASH PROOFING)
        text = self.get_val(kwargs, "text", "", str)
        seed = self.get_val(kwargs, "seed", 0, int)

        # ============================================================
        # CORE PROCESSING
        # ============================================================
        
        # Strip comments
        protected_text = text.replace('__#', '___UMI_HASH_PROTECT___').replace('<#', '<___UMI_HASH_PROTECT___')
        clean_lines = []
        for line in protected_text.splitlines():
            if '//' in line:
                line = line.split('//')[0]
            if '#' in line:
                line = line.split('#')[0]
            line = line.strip()
            if line:
                clean_lines.append(line)
        
        text = "\n".join(clean_lines)
        text = text.replace('___UMI_HASH_PROTECT___', '#').replace('<___UMI_HASH_PROTECT___', '<#')

        random.seed(seed)
        
        options = {
            'verbose': False, 
            'seed': seed,
            'ignore_paths': True
        }

        # Fix: Get all paths instead of just the internal one
        all_wildcard_paths = get_all_wildcard_paths()
        tag_loader = TagLoader(all_wildcard_paths, options)
        
        tag_selector = TagSelector(tag_loader, options)
        tag_replacer = TagReplacer(tag_selector)
        dynamic_replacer = DynamicPromptReplacer(seed)
        conditional_replacer = ConditionalReplacer()
        variable_replacer = VariableReplacer()

        globals_dict = tag_loader.load_globals()
        variable_replacer.load_globals(globals_dict)

        prompt = text
        previous_prompt = ""
        iterations = 0
        tag_selector.clear_seeded_values()

        while previous_prompt != prompt and iterations < 50:
            previous_prompt = prompt
            prompt = variable_replacer.store_variables(prompt, tag_replacer, dynamic_replacer)
            tag_selector.update_variables(variable_replacer.variables)
            prompt = variable_replacer.replace_variables(prompt)
            prompt = tag_replacer.replace(prompt)
            prompt = dynamic_replacer.replace(prompt)
            iterations += 1
            
        # FIX: PASS VARIABLES TO CONDITIONAL REPLACER
        prompt = conditional_replacer.replace(prompt, variable_replacer.variables)
        
        additions = tag_selector.get_prefixes_and_suffixes()
        if additions['prefixes']:
            prompt = ", ".join(additions['prefixes']) + ", " + prompt
        if additions['suffixes']:
            prompt = prompt + ", " + ", ".join(additions['suffixes'])

        # Clean up formatting
        prompt = re.sub(r',\s*,', ',', prompt)
        prompt = re.sub(r'\s+', ' ', prompt).strip().strip(',')

        return (prompt,)

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@server.PromptServer.instance.routes.get("/ab_wildcard/wildcards")
async def get_wildcards(request):
    all_paths = get_all_wildcard_paths()
    options = {'ignore_paths': True, 'verbose': False}
    loader = TagLoader(all_paths, options)
    loader.build_index()
    
    # Get Wildcards
    wildcards = sorted(list(loader.files_index))
    
    return web.json_response({
        "wildcards": wildcards
    })

@server.PromptServer.instance.routes.post("/ab_wildcard/refresh")
async def refresh_wildcards(request):
    """Refreshes the global cache and returns the new list."""
    GLOBAL_CACHE.clear()
    
    # Fix: Nuke the Index Cache so it rebuilds next time
    GLOBAL_INDEX['built'] = False 
    GLOBAL_INDEX['files'] = set()
    GLOBAL_INDEX['entries'] = {}
    
    all_paths = get_all_wildcard_paths()
    options = {'ignore_paths': True, 'verbose': False}
    loader = TagLoader(all_paths, options)
    loader.build_index() # Rebuild immediately
    
    wildcards = sorted(list(loader.files_index))
    
    return web.json_response({
        "status": "success", 
        "count": len(wildcards),
        "wildcards": wildcards
    })
