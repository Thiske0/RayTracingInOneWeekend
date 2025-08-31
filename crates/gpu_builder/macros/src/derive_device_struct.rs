use crate::{
    common::{device_generics_from_host, require_repr},
    derive_device_impl::{derive_device_impl_inner, device_impl_attributes::DeviceImplAttributes},
};
use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, parse_quote, punctuated::Punctuated, token::Comma, Attribute, Fields,
    GenericArgument, GenericParam, Generics, Ident, Item, ItemEnum, ItemStruct, Lifetime,
    PathArguments, Result, Type,
};

pub fn derive_device_struct(
    _attrs: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as Item);
    let impl_block = derive_device_struct_inner(input);
    impl_block.unwrap_or_else(|e| e.to_compile_error()).into()
}

pub fn derive_device_struct_inner(input: Item) -> syn::Result<TokenStream> {
    #[allow(unused)]
    let struct_impl = match &input {
        Item::Struct(s) => derive_device_struct_struct(s)?,
        Item::Enum(e) => derive_device_struct_enum(e)?,
        _ => Err(syn::Error::new_spanned(
            input.clone(),
            "#[derive_device_struct] only works for structs or enums",
        ))?,
    };
    let parsed_struct_impl = syn::parse2::<Item>(struct_impl)?;
    let device_impl =
        derive_device_impl_inner(create_device_impl_attributes(&input)?, &parsed_struct_impl)?;
    let fixed_input = replace_host_only(input);
    Ok(quote! {
        #fixed_input
        #parsed_struct_impl
        #device_impl
    })
}

fn replace_host_only(mut input: Item) -> TokenStream {
    if let Item::Struct(item) = &mut input {
        // replace all #[host_only] fields with #[cfg(not(target_os="cuda"))]
        for field in item.fields.iter_mut() {
            field.attrs.retain(|attr| !attr.path().is_ident("no_copy"));
            if field
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("host_only"))
            {
                field
                    .attrs
                    .retain(|attr| !attr.path().is_ident("host_only"));
                field.attrs.push(parse_quote! {
                    #[cfg(not(target_os = "cuda"))]
                });
            }
        }
    }
    quote! {
        #input
    }
}

fn create_device_impl_attributes(input: &Item) -> syn::Result<DeviceImplAttributes> {
    match input {
        Item::Struct(s) => create_device_impl_attributes_struct(s),
        Item::Enum(e) => create_device_impl_attributes_enum(e),
        _ => unreachable!(),
    }
}

fn create_device_impl_attributes_struct(
    input: &syn::ItemStruct,
) -> syn::Result<DeviceImplAttributes> {
    let builder = input.ident.clone();
    let (lifetime, extra_lifetime) = get_lifetimes(&input.generics)?;
    Ok(DeviceImplAttributes {
        builder,
        lifetime,
        extra_lifetime,
    })
}

fn create_device_impl_attributes_enum(input: &syn::ItemEnum) -> syn::Result<DeviceImplAttributes> {
    let builder = input.ident.clone();
    let (lifetime, extra_lifetime) = get_lifetimes(&input.generics)?;
    Ok(DeviceImplAttributes {
        builder,
        lifetime,
        extra_lifetime,
    })
}

fn get_lifetimes(generics: &Generics) -> Result<(Option<Lifetime>, Vec<Lifetime>)> {
    let mut life_time: Option<Lifetime> = None;
    let mut extra_lifetimes = Vec::new();
    for param in generics.params.iter() {
        if let syn::GenericParam::Lifetime(lt) = param {
            extra_lifetimes.push(lt.lifetime.clone());
        } else if let GenericParam::Type(ty_param) = param {
            for bound in &ty_param.bounds {
                if let syn::TypeParamBound::Trait(trait_bound) = bound {
                    for path_segment in &trait_bound.path.segments {
                        if path_segment.ident == "gpu_builder" {
                            continue;
                        } else if path_segment.ident == "Builder" {
                            if let syn::PathArguments::AngleBracketed(args) =
                                &path_segment.arguments
                            {
                                if args.args.len() != 1 {
                                    return Err(syn::Error::new_spanned(
                                        args,
                                        "Builder must have exactly one lifetime argument",
                                    ));
                                } else if let syn::GenericArgument::Lifetime(builder_lifetime) =
                                    &args.args[0]
                                {
                                    if let Some(lifetime) = &life_time {
                                        if lifetime.ident != builder_lifetime.ident {
                                            return Err(syn::Error::new_spanned(
                                                args,
                                                "Builders must have the same lifetime argument",
                                            ));
                                        }
                                    } else {
                                        life_time = Some(builder_lifetime.clone());
                                    }
                                }
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }
    if let Some(lifetime) = &life_time {
        extra_lifetimes.retain(|lt| lt.ident != lifetime.ident);
    }
    Ok((life_time, extra_lifetimes))
}

fn filter_attributes(attributes: Vec<Attribute>) -> Vec<Attribute> {
    attributes
        .into_iter()
        .filter(|attr| !attr.path().is_ident("enum_dispatch"))
        .collect()
}

fn derive_device_struct_struct(input: &ItemStruct) -> Result<TokenStream> {
    require_repr(&input.attrs)?;
    let attributes = filter_attributes(input.attrs.clone());
    let visibility = &input.vis;
    let device_name = Ident::new(&format!("{}Device", input.ident), Span::mixed_site());

    let (device_generics, changed_types) = device_generics_from_host(&input.generics);
    let device_where_clause = device_generics.where_clause.clone();

    let fields = match &input.fields {
        Fields::Named(fields) => Ok(&fields.named),
        _ => Err(syn::Error::new_spanned(
            input.fields.clone(),
            "#[derive_device_struct] for structs only works with named fields",
        )),
    }?;

    let fields: Punctuated<_, Comma> = fields
        .iter()
        .filter(|field| {
            for attr in &field.attrs {
                if attr.path().is_ident("host_only") || attr.path().is_ident("no_copy") {
                    return false;
                }
            }
            true
        })
        .collect();

    let struct_fields = fields.iter().map(|field| {
        let field_name = &field.ident;
        let device_field_ty = device_field_type(&field.ty, &changed_types);
        quote! {
            #field_name: #device_field_ty
        }
    });

    let impl_block = quote! {
        #(#attributes)*
        #visibility struct #device_name #device_generics #device_where_clause {
            #(#struct_fields),*
        }
    };
    Ok(impl_block)
}

fn derive_device_struct_enum(input: &ItemEnum) -> Result<TokenStream> {
    require_repr(&input.attrs)?;
    let attributes = filter_attributes(input.attrs.clone());
    let visibility = &input.vis;
    let device_name = Ident::new(&format!("{}Device", input.ident), Span::mixed_site());

    let (device_generics, changed_types) = device_generics_from_host(&input.generics);
    let device_where_clause = device_generics.where_clause.clone();

    let variants = input
        .variants
        .iter()
        .map(|variant| {
            let variant_name = &variant.ident;
            let variant_index = &variant.discriminant.as_ref();
            let variant_index = variant_index.map(|(_, expr)| {
                quote! { = #expr }
            });
            match &variant.fields {
                Fields::Named(fields) => {
                    let fields = &fields.named;
                    let field_types = fields.iter().map(|f| {
                        let name = &f.ident;
                        let ty = &f.ty;
                        let device_ty = device_field_type(ty, &changed_types);
                        quote! { #name: #device_ty }
                    });
                    quote! {
                        #variant_name { #(#field_types),* } #variant_index
                    }
                }
                Fields::Unnamed(fields) => {
                    let fields = &fields.unnamed;
                    let field_types = fields.iter().map(|f| {
                        let ty = &f.ty;
                        let device_ty = device_field_type(ty, &changed_types);
                        quote! { #device_ty }
                    });
                    quote! {
                        #variant_name ( #(#field_types),* ) #variant_index
                    }
                }
                Fields::Unit => quote! {
                    #variant_name #variant_index
                },
            }
        })
        .collect::<Vec<_>>();

    let impl_block = quote! {
        #(#attributes)*
        #visibility enum #device_name #device_generics #device_where_clause {
            #(#variants),*
        }
    };
    Ok(impl_block)
}

fn device_field_type(field_ty: &Type, changed_types: &[Type]) -> TokenStream {
    let changed_types_tokens = changed_types
        .iter()
        .map(|ty| ty.to_token_stream().to_string())
        .collect::<Vec<_>>();
    if changed_types_tokens.contains(&field_ty.to_token_stream().to_string()) {
        quote! {
            #field_ty
        }
    } else {
        if let Type::Reference(type_reference) = field_ty {
            let elem = &type_reference.elem;
            quote! {
               <&'static #elem as gpu_builder::Builder<'static>>::Output
            }
        } else if let Type::Ptr(type_ptr) = field_ty {
            let elem = &type_ptr.elem;
            let ptr = if type_ptr.mutability.is_some() {
                quote! { *mut }
            } else {
                quote! { *const }
            };
            let elem_ty = device_field_type(elem, changed_types);
            quote! {
               #ptr #elem_ty
            }
        } else if let Type::Path(path) = field_ty {
            if path.path.segments.iter().any(|segment| {
                if let PathArguments::AngleBracketed(args) = &segment.arguments {
                    args.args.iter().any(|arg| {
                        if let GenericArgument::Lifetime(_lifetime) = arg {
                            true
                        } else {
                            false
                        }
                    })
                } else {
                    false
                }
            }) {
                //covert Ident to device version and remove all lifetime arguments
                let mut new_type = path.clone();
                if path.path.segments.len() != 1 {
                    panic!("Only single segment types with lifetimes are supported");
                }
                new_type.path.segments[0].ident = Ident::new(
                    &format!("{}Device", path.path.segments[0].ident),
                    Span::mixed_site(),
                );
                if let PathArguments::AngleBracketed(ref mut args) =
                    new_type.path.segments[0].arguments
                {
                    args.args = args
                        .args
                        .clone()
                        .into_iter()
                        .filter(|arg| {
                            if let GenericArgument::Lifetime(_lifetime) = arg {
                                false
                            } else {
                                true
                            }
                        })
                        .map(|mut arg| {
                            if let GenericArgument::Type(ref mut ty) = arg {
                                let device_type = device_field_type(ty, changed_types);
                                *ty = parse_quote! { #device_type };
                            }
                            arg
                        })
                        .collect();
                } else {
                    panic!("Expected angle bracketed arguments");
                };
                quote! {
                   #new_type
                }
            } else {
                quote! {
                    <#path as gpu_builder::Builder<'static>>::Output
                }
            }
        } else {
            quote! {
                <#field_ty as gpu_builder::Builder<'static>>::Output
            }
        }
    }
}
